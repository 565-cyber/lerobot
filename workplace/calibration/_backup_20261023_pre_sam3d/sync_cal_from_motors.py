#!/usr/bin/env python
"""把电机 EEPROM 里的标定读出来，覆盖 LeRobot 缓存里的标定 JSON。

背景（2026-10-20）：
    `lerobot-record` 启动时报
      "Mismatch between calibration values in the motor and the calibration file"
    实测两臂 EEPROM 的 homing_offset / range 与
    ~/.cache/huggingface/lerobot/calibration/ 下的 JSON 在**每个关节**都不一致。

    证据表明 EEPROM 才是当前这对手臂的真实标定：
      - 两臂 EEPROM 的 wrist_roll 行程几乎相同（181–4060 / 233–4041），
        而两个 JSON 的 wrist_roll 互相差 ~1000 counts；
      - `workplace/calibration/arm.py` 里记录的实测行程
        (shoulder_lift 207.9°, wrist_flex 200.0°, wrist_roll 340.9°)
        与 EEPROM 完全吻合，与 JSON 不吻合；
      - follower JSON 的 wrist_roll range_max=4940 超出 12 位编码器范围。

    因此方向是 **JSON ← EEPROM**，而不是提示里默认的 EEPROM ← JSON。
    覆盖前会自动备份。

用法（先确保 lerobot-record / teleoperate 已退出，串口没被占用）：
    /home/wzy/anaconda3/envs/lerobot/bin/python sync_cal_from_motors.py
"""

import json
import shutil
import time
from pathlib import Path

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
MOTORS = {n: Motor(i, "sts3215", MotorNormMode.RANGE_M100_100) for i, n in enumerate(NAMES, start=1)}

CACHE = Path.home() / ".cache/huggingface/lerobot/calibration"
TARGETS = {
    "follower": (
        "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14144471-if00",
        CACHE / "robots/so101_follower/my_awesome_follower_arm.json",
    ),
    "leader": (
        "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14144911-if00",
        CACHE / "teleoperators/so101_leader/my_awesome_leader_arm.json",
    ),
}


def as_json(cal: dict[str, MotorCalibration]) -> dict:
    return {
        n: {
            "id": c.id,
            "drive_mode": c.drive_mode,
            "homing_offset": c.homing_offset,
            "range_min": c.range_min,
            "range_max": c.range_max,
        }
        for n, c in cal.items()
    }


def main() -> int:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for role, (port, path) in TARGETS.items():
        bus = FeetechMotorsBus(port=port, motors=MOTORS)
        bus.connect()
        try:
            eeprom = as_json(bus.read_calibration())
            old = json.loads(path.read_text()) if path.exists() else {}
            print(f"== {role}: {path}")
            for n in NAMES:
                o, e = old.get(n, {}), eeprom[n]
                delta = e["homing_offset"] - o.get("homing_offset", 0) if o else None
                print(
                    f"   {n:14s} off {o.get('homing_offset'):>7} -> {e['homing_offset']:>7}"
                    f"   range [{o.get('range_min')},{o.get('range_max')}]"
                    f" -> [{e['range_min']},{e['range_max']}]"
                    f"   (d_off={delta})"
                )
            backup = path.with_suffix(path.suffix + f".bak-{stamp}")
            if path.exists():
                shutil.copy2(path, backup)
                print(f"   backup: {backup}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(eeprom, indent=4, ensure_ascii=False) + "\n")
            print(f"   written: {path}")
        finally:
            bus.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
