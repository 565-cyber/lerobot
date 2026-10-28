#!/usr/bin/env python
"""
LeIsaac Environment Quick Test

Usage:
  # Headless mode (no window, fast test)
  python examples/leisaac/test_env.py

  # With rendering window (need DISPLAY, e.g. ssh -X)
  python examples/leisaac/test_env.py --no-headless

  # With livestream (watch in browser at http://<ip>:8211)
  python examples/leisaac/test_env.py --livestream

  # Choose environment
  python examples/leisaac/test_env.py --env envs/so101_pick_orange.py
"""

import argparse
import os
import sys
import time

os.environ.setdefault("ACCEPT_EULA", "Y")
os.environ.setdefault("PRIVACY_CONSENT", "Y")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")


def test_imports_only():
    """Test that all Python imports work (no SimulationApp needed)."""
    print("=" * 50)
    print("Testing import chain...")
    print("=" * 50)

    checks = []

    # These should work WITHOUT SimulationApp
    for name, import_stmt in [
        ("lerobot.envs.factory", "from lerobot.envs.factory import make_env"),
        ("lerobot", "import lerobot"),
    ]:
        try:
            exec(import_stmt)
            checks.append((name, True, None))
        except Exception as e:
            checks.append((name, False, str(e)))

    for name, ok, err in checks:
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {name}")
        if err:
            print(f"         {err}")

    all_ok = all(ok for _, ok, _ in checks)
    print(f"Result: {'ALL PASSED' if all_ok else 'SOME FAILED'}\n")
    return all_ok


def test_environment(env_file="envs/so101_lift_cube.py", num_steps=50,
                     headless=True, livestream=False):
    """
    Load and run a LeIsaac environment.

    This calls make_env() which internally:
      1. Downloads env.py + assets from HuggingFace Hub
      2. Initializes SimulationApp via AppLauncher
      3. Creates the Gym environment
    """
    print("=" * 50)
    print(f"Loading: {env_file}")
    print(f"Mode: {'headless' if headless else 'windowed'}"
          f"{' + livestream' if livestream else ''}")
    print("=" * 50)

    import torch
    from lerobot.envs.factory import make_env

    hub_path = f"LightwheelAI/leisaac_env:{env_file}"

    try:
        t0 = time.time()
        print("Initializing SimulationApp + Environment...")
        envs_dict = make_env(hub_path, n_envs=1, trust_remote_code=True)
        print(f"  Loaded in {time.time() - t0:.1f}s")

        suite_name = next(iter(envs_dict))
        env = envs_dict[suite_name][0].envs[0].unwrapped

        print(f"  Type: {type(env).__name__}")
        print(f"  Action space: {env.action_space}")

        # Reset
        print("Resetting...")
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        print(f"  Observation keys: {list(obs.keys()) if isinstance(obs, dict) else type(obs)}")

        # Run steps
        print(f"Running {num_steps} steps...")
        t0 = time.time()
        for i in range(num_steps):
            action = torch.tensor(env.action_space.sample())
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                env.reset()

            if (i + 1) % 25 == 0:
                elapsed = time.time() - t0
                print(f"  Step {i + 1}/{num_steps} | FPS: {(i + 1) / elapsed:.1f}")

        elapsed = time.time() - t0
        print(f"  Done: {num_steps} steps in {elapsed:.1f}s ({num_steps / elapsed:.1f} FPS)")

        env.close()
        print("[OK] Environment test passed!\n")
        return True

    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="LeIsaac Environment Test")
    parser.add_argument("--env", type=str, default="envs/so101_lift_cube.py")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--no-headless", action="store_true",
                        help="Show rendering window (needs DISPLAY)")
    parser.add_argument("--livestream", action="store_true",
                        help="Enable livestream at http://<ip>:8211")
    parser.add_argument("--imports-only", action="store_true",
                        help="Only test imports, skip environment")
    args = parser.parse_args()

    print("LeIsaac Environment Test")
    print("=" * 50)

    # Step 1: Test imports (no SimulationApp)
    if not test_imports_only():
        sys.exit(1)

    if args.imports_only:
        print("Import test complete. Skipping environment test.")
        return

    # Step 2: Run environment
    headless = not args.no_headless
    env_ok = test_environment(
        args.env, args.steps,
        headless=headless,
        livestream=args.livestream,
    )

    if not env_ok:
        print("\nEnvironment test FAILED.")
        sys.exit(1)

    print("=" * 50)
    print("ALL TESTS PASSED")
    print("=" * 50)
    print(f"\nYour LeIsaac simulation is ready!")
    print(f"\nTo see the simulation画面:")
    if args.no_headless:
        print("  Already running with rendering window.")
    else:
        print("  Option A: ssh -X to this machine, then run with --no-headless")
        print("  Option B: Run with --livestream and open http://<ip>:8211 in browser")
        print("  Option C: Record data headlessly, then replay on a machine with display")


if __name__ == "__main__":
    main()
