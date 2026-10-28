#!/usr/bin/env python
"""
LeIsaac Simulation Example: Teleoperation & Data Recording

This script demonstrates how to use teleoperation with a LeIsaac environment
and record data in LeRobot format.

Requirements:
  - SO101 Leader arm connected via USB (/dev/ttyACM0)
  - Or keyboard teleoperation (no hardware needed)

Usage:
  # With SO101 Leader arm:
  python examples/leisaac/02_teleop_record.py --teleop so101leader --port /dev/ttyACM0

  # With keyboard (no hardware needed):
  python examples/leisaac/02_teleop_record.py --teleop keyboard

  # With recording enabled:
  python examples/leisaac/02_teleop_record.py --teleop keyboard --record
"""

import argparse
import os
import sys

os.environ.setdefault("ACCEPT_EULA", "Y")
os.environ.setdefault("PRIVACY_CONSENT", "Y")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")


def main():
    parser = argparse.ArgumentParser(description="LeIsaac Teleoperation Example")
    parser.add_argument(
        "--env",
        type=str,
        default="envs/so101_lift_cube.py",
        help="Environment file from LightwheelAI/leisaac_env",
    )
    parser.add_argument(
        "--teleop",
        type=str,
        default="keyboard",
        choices=["so101leader", "keyboard", "gamepad", "bi-so101leader"],
        help="Teleoperation device type",
    )
    parser.add_argument(
        "--port",
        type=str,
        default="/dev/ttyACM0",
        help="Serial port for leader arm",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Enable data recording",
    )
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        default="my_username/leisaac_dataset",
        help="Dataset repository ID for recording",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Recording FPS",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=5,
        help="Number of episodes to record",
    )
    args = parser.parse_args()

    import torch
    from lerobot.envs.factory import make_env

    hub_path = f"LightwheelAI/leisaac_env:{args.env}"
    print(f"Loading environment: {hub_path}")
    print(f"Teleop device: {args.teleop}")

    envs_dict = make_env(hub_path, n_envs=1, trust_remote_code=True)
    suite_name = next(iter(envs_dict))
    sync_vector_env = envs_dict[suite_name][0]
    env = sync_vector_env.envs[0].unwrapped

    print(f"Environment loaded: {type(env).__name__}")

    # Setup data recording if requested
    if args.record:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.datasets.utils import build_dataset_frame
        from lerobot.datasets.pipeline_features import (
            aggregate_pipeline_dataset_features,
            create_initial_features,
            combine_feature_dicts,
        )
        from lerobot.utils.constants import ACTION, OBS_STR

        print(f"Recording to dataset: {args.dataset_repo_id}")
        dataset = LeRobotDataset.create(
            args.dataset_repo_id,
            args.fps,
            robot_type="so101_follower",
            use_videos=True,
        )
    else:
        dataset = None

    # Main teleoperation loop
    print("\n" + "=" * 60)
    print("CONTROLS:")
    if args.teleop == "keyboard":
        print("  W/S: Move end-effector forward/backward")
        print("  A/D: Move end-effector left/right")
        print("  Q/E: Move end-effector up/down")
        print("  I/K: Rotate pitch")
        print("  J/L: Rotate yaw")
        print("  U/O: Rotate roll")
        print("  Space: Toggle gripper")
        print("  R: Reset environment")
        print("  ESC: Exit")
    else:
        print("  Use the leader arm to control the robot")
        print("  Press 'b' to begin teleoperation")
        print("  Press 'n' for success reset, 'r' for failure reset")
    print("=" * 60 + "\n")

    obs, info = env.reset()
    episode = 0
    step = 0

    try:
        while episode < args.num_episodes:
            # For keyboard teleop, read keyboard input
            if args.teleop == "keyboard":
                action = _get_keyboard_action(env)
            else:
                # For leader arm, action comes from teleop device
                action = torch.tensor(env.action_space.sample())  # placeholder

            obs, reward, terminated, truncated, info = env.step(action)
            step += 1

            if dataset is not None:
                # Record frame
                frame = {
                    **{"observation.state": obs.get("observation.state", obs.get("state"))},
                    "action": action.numpy(),
                    "task": args.env,
                }
                dataset.add_frame(frame)

            if terminated or truncated:
                episode += 1
                print(f"Episode {episode}/{args.num_episodes} completed ({step} steps)")
                if dataset is not None:
                    dataset.save_episode()
                obs, info = env.reset()
                step = 0

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    env.close()
    if dataset is not None:
        dataset.finalize()
        print(f"Dataset saved: {args.dataset_repo_id}")

    print("Done!")


def _get_keyboard_action(env):
    """Read keyboard input and convert to action."""
    # This is a simplified placeholder. In practice, use the leisaac
    # keyboard teleop device for proper control.
    import torch
    return torch.tensor(env.action_space.sample())


if __name__ == "__main__":
    main()
