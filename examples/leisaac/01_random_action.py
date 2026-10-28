#!/usr/bin/env python
"""
LeIsaac Simulation Example: Random Action

This script demonstrates how to load a LeIsaac environment from HuggingFace Hub
and run it with random actions.

Available environments:
  - envs/so101_pick_orange.py   (Pick up an orange)
  - envs/so101_lift_cube.py     (Lift a cube)
  - envs/so101_clean_toytable.py (Clean toy table, single arm)
  - envs/bi_so101_fold_cloth.py (Fold cloth, dual arm)

Usage:
  python examples/leisaac/01_random_action.py
  python examples/leisaac/01_random_action.py --env envs/so101_pick_orange.py
  python examples/leisaac/01_random_action.py --headless False
"""

import argparse
import os
import sys
import time

# Ensure the environment is properly set up before importing leisaac modules
os.environ.setdefault("ACCEPT_EULA", "Y")
os.environ.setdefault("PRIVACY_CONSENT", "Y")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

# Use HF mirror for users in China
if "HF_ENDPOINT" not in os.environ:
    # Uncomment the line below if you need to use a mirror:
    # os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    pass


def main():
    parser = argparse.ArgumentParser(description="LeIsaac Random Action Example")
    parser.add_argument(
        "--env",
        type=str,
        default="envs/so101_lift_cube.py",
        help="Environment file to load from LightwheelAI/leisaac_env",
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Show rendering window (needs display)",
    )
    parser.add_argument(
        "--enable-cameras",
        type=bool,
        default=True,
        help="Enable camera sensors",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=300,
        help="Number of simulation steps",
    )
    args = parser.parse_args()

    import torch
    from lerobot.envs.factory import make_env

    hub_path = f"LightwheelAI/leisaac_env:{args.env}"
    print(f"Loading environment: {hub_path}")
    print(f"Headless: {not args.no_headless}, Cameras: {args.enable_cameras}")

    # Load the environment via LeRobot EnvHub
    envs_dict = make_env(hub_path, n_envs=1, trust_remote_code=True)

    suite_name = next(iter(envs_dict))
    sync_vector_env = envs_dict[suite_name][0]
    env = sync_vector_env.envs[0].unwrapped

    print(f"Environment loaded: {type(env).__name__}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    # Reset and run
    obs, info = env.reset()
    print("Environment reset. Starting simulation loop...")

    start_time = time.perf_counter()
    episode_reward = 0.0
    episode_count = 0

    for step in range(args.num_steps):
        # Sample random action from the action space
        action = torch.tensor(env.action_space.sample())

        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += float(reward) if reward is not None else 0.0

        if terminated or truncated:
            episode_count += 1
            print(
                f"Episode {episode_count} ended at step {step + 1}. "
                f"Reward: {episode_reward:.3f}"
            )
            obs, info = env.reset()
            episode_reward = 0.0

        if (step + 1) % 100 == 0:
            elapsed = time.perf_counter() - start_time
            fps = (step + 1) / elapsed
            print(f"Step {step + 1}/{args.num_steps} | FPS: {fps:.1f}")

    elapsed = time.perf_counter() - start_time
    print(f"Completed {args.num_steps} steps in {elapsed:.1f}s ({args.num_steps / elapsed:.1f} FPS)")
    env.close()
    print("Done!")


if __name__ == "__main__":
    main()
