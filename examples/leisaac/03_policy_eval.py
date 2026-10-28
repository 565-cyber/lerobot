#!/usr/bin/env python
"""
LeIsaac Simulation Example: Policy Evaluation

This script demonstrates how to evaluate a trained LeRobot policy in a
LeIsaac simulation environment.

Usage:
  # With a local policy checkpoint:
  python examples/leisaac/03_policy_eval.py \\
      --policy.path=/path/to/your/policy/checkpoint \\
      --env=envs/so101_lift_cube.py

  # With a HuggingFace Hub policy:
  python examples/leisaac/03_policy_eval.py \\
      --policy.path=your_username/your_policy \\
      --env=envs/so101_pick_orange.py

  # With visualization (non-headless):
  python examples/leisaac/03_policy_eval.py \\
      --policy.path=/path/to/policy \\
      --env=envs/so101_lift_cube.py \\
      --headless=False
"""

import argparse
import os
import sys
import time

# Use local lerobot code instead of pip-installed 0.4.1 (has dataclass bug)
sys.path.insert(0, "/home/zjq/lerobot/src")

os.environ.setdefault("ACCEPT_EULA", "Y")
os.environ.setdefault("PRIVACY_CONSENT", "Y")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")


def main():
    parser = argparse.ArgumentParser(description="LeIsaac Policy Evaluation")
    parser.add_argument(
        "--policy.path",
        type=str,
        required=True,
        help="Path to policy checkpoint (local path or HuggingFace repo ID)",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="envs/so101_lift_cube.py",
        help="Environment file from LightwheelAI/leisaac_env",
    )
    parser.add_argument(
        "--headless",
        type=bool,
        default=True,
        help="Run in headless mode",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for policy inference (cuda/cpu)",
    )
    args = parser.parse_args()

    import torch
    from lerobot.envs.factory import make_env
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_policy
    from lerobot.policies.utils import (
        make_robot_action,
        prepare_observation_for_inference,
    )
    from lerobot.processor import make_default_processors
    from lerobot.utils.constants import ACTION, OBS_STR

    # Load environment
    hub_path = f"LightwheelAI/leisaac_env:{args.env}"
    print(f"Loading environment: {hub_path}")
    envs_dict = make_env(hub_path, n_envs=1, trust_remote_code=True)
    suite_name = next(iter(envs_dict))
    sync_vector_env = envs_dict[suite_name][0]
    env = sync_vector_env.envs[0].unwrapped

    # Load policy
    policy_path = getattr(args, "policy.path")
    print(f"Loading policy from: {policy_path}")
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = policy_path
    policy_cfg.device = args.device
    policy = make_policy(policy_cfg)
    print(f"Policy type: {type(policy).__name__}")
    print(f"Device: {args.device}")

    # Setup processors
    _, robot_action_processor, robot_observation_processor = make_default_processors()

    policy.reset()

    # Evaluation loop
    total_rewards = []
    total_steps = 0
    start_time = time.perf_counter()

    for episode in range(args.num_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs, info = obs

        episode_reward = 0.0

        for step in range(args.max_steps):
            # Process observation
            obs_processed = robot_observation_processor(obs)

            # Prepare for policy inference
            observation = prepare_observation_for_inference(
                obs_processed,
                device=args.device,
                task="leisaac_task",
                robot_type="so101_follower",
            )

            # Get action from policy
            with torch.inference_mode():
                action_values = policy.predict_action(observation)
                action_values = policy.postprocessor(action_values)

            # Send action to environment
            action_tensor = action_values.squeeze(0).cpu()
            obs, reward, terminated, truncated, info = env.step(action_tensor)
            if isinstance(obs, tuple):
                obs, info = obs

            if reward is not None:
                episode_reward += float(reward)

            if terminated or truncated:
                break

        total_rewards.append(episode_reward)
        total_steps += step + 1

        print(
            f"Episode {episode + 1}/{args.num_episodes} | "
            f"Steps: {step + 1} | Reward: {episode_reward:.3f}"
        )

    elapsed = time.perf_counter() - start_time
    avg_reward = sum(total_rewards) / len(total_rewards)
    success_rate = sum(1 for r in total_rewards if r > 0) / len(total_rewards)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Episodes:        {args.num_episodes}")
    print(f"Avg reward:      {avg_reward:.3f}")
    print(f"Success rate:    {success_rate:.1%}")
    print(f"Avg steps/ep:    {total_steps / args.num_episodes:.1f}")
    print(f"Total time:      {elapsed:.1f}s")
    print(f"Avg time/ep:     {elapsed / args.num_episodes:.1f}s")
    print("=" * 50)

    env.close()
    print("Done!")


if __name__ == "__main__":
    main()
