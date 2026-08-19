# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import mujoco


def load_joint_limits(model_path: str | Path) -> dict[str, tuple[float, float]]:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    limits: dict[str, tuple[float, float]] = {}
    for jid in range(model.njnt):
        name = model.joint(jid).name
        limits[name] = (float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1]))
    return limits


def clip_pose(pose: dict[str, float], limits: dict[str, tuple[float, float]]) -> dict[str, float]:
    clipped = {}
    for name, value in pose.items():
        lo, hi = limits[name]
        clipped[name] = max(lo, min(hi, float(value)))
    return clipped

