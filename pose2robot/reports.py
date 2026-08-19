# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from .constants import LM, ROBOT_JOINTS


def upper_body_visibility(frame: dict) -> float:
    names = ["nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"]
    return float(np.mean([frame["pts"][LM[name]][3] for name in names]))


def write_joint_csv(path: str | Path, rows: list[dict]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return out


def write_joint_report(path: str | Path, pose_data: dict, rows: list[dict], limits: dict[str, tuple[float, float]]) -> Path:
    low_visibility = sum(1 for frame in pose_data["landmarks"] if upper_body_visibility(frame) < 0.3)
    lines = [
        f"# v2 机器人关节映射报告: {Path(path).stem}",
        "",
        f"- 输入帧数: {len(rows)}",
        f"- 视频 FPS: {float(pose_data.get('fps', 0.0)):.2f}",
        f"- 上半身低置信帧: {low_visibility} / {len(rows)}",
        "",
        "| 关节 | min | max | mean | limit_hits |",
        "|---|---:|---:|---:|---:|",
    ]
    for joint in ROBOT_JOINTS:
        values = np.asarray([row[joint] for row in rows], dtype=float)
        lo, hi = limits[joint]
        hits = int(np.sum((values <= lo + 1e-5) | (values >= hi - 1e-5)))
        lines.append(f"| `{joint}` | {values.min():+.3f} | {values.max():+.3f} | {values.mean():+.3f} | {hits} |")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def check_motion(motion: dict, limits: dict[str, tuple[float, float]]) -> list[str]:
    errors = []
    previous_t = -math.inf
    for idx, keyframe in enumerate(motion.get("keyframes", [])):
        t = keyframe.get("t")
        if not isinstance(t, (int, float)) or not math.isfinite(float(t)):
            errors.append(f"kf[{idx}] t 非法: {t}")
            continue
        if float(t) < previous_t:
            errors.append(f"kf[{idx}] t 倒退: {t} < {previous_t}")
        previous_t = float(t)
        for joint, value in keyframe.get("pose", {}).items():
            if joint not in limits:
                errors.append(f"kf[{idx}] 未知关节: {joint}")
                continue
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                errors.append(f"kf[{idx}] {joint} 非有限数: {value}")
                continue
            lo, hi = limits[joint]
            if float(value) < lo - 1e-6 or float(value) > hi + 1e-6:
                errors.append(f"kf[{idx}] {joint} 超限: {value:.3f} not in [{lo:.3f}, {hi:.3f}]")
    if not motion.get("keyframes"):
        errors.append("Motion 没有 keyframes")
    return errors

