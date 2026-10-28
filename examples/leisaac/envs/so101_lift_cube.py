"""
Local copy of LightwheelAI/leisaac_env envs/so101_lift_cube.py.
Modified to support headless mode via HEADLESS env var.
"""
import os
from pathlib import Path
from huggingface_hub import snapshot_download


def make_env(n_envs: int = 1, use_async_envs: bool = False):
    # download assets from huggingface first
    snapshot_path = snapshot_download(
        repo_id='LightwheelAI/leisaac_env', revision=None, cache_dir=None,
    )

    # set environment variable for leisaac asset
    assets_root = Path(snapshot_path) / 'assets'
    os.environ['LEISAAC_ASSETS_ROOT'] = str(assets_root)

    from isaaclab.app import AppLauncher

    app_config = {"enable_cameras": True}
    if os.environ.get("HEADLESS", "").lower() in ("1", "true", "yes"):
        app_config["headless"] = True

    _ = AppLauncher(app_config)

    from leisaac.utils.envhub_utils import export_env

    return export_env(
        'LeIsaac-SO101-LiftCube-v0', device="cuda:0", num_envs=n_envs
    )
