# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image

from .video import sample_indices


def _camera() -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[0] = 0.0
    cam.lookat[1] = 0.0
    cam.lookat[2] = -0.15
    cam.distance = 3.0
    cam.elevation = -8.0
    cam.azimuth = 135.0
    return cam


def _apply_pose(model: mujoco.MjModel, data: mujoco.MjData, pose: dict[str, float]) -> None:
    data.qpos[:] = model.qpos0
    for name, value in pose.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"未知关节: {name}")
        adr = int(model.joint(jid).qposadr[0])
        lo = float(model.jnt_range[jid][0])
        hi = float(model.jnt_range[jid][1])
        data.qpos[adr] = max(lo, min(hi, float(value)))


def _render_pose(model, data, renderer, cam, pose: dict[str, float]) -> np.ndarray:
    _apply_pose(model, data, pose)
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=cam)
    return renderer.render()


class LiveRobotRenderer:
    def __init__(self, model_path: str | Path):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=480, width=480)
        self.cam = _camera()

    def render(self, pose: dict[str, float]) -> np.ndarray:
        return _render_pose(self.model, self.data, self.renderer, self.cam, pose)

    def close(self) -> None:
        self.renderer.close()


def render_motion_samples(
    motion: dict,
    model_path: str | Path,
    out_dir: str | Path,
    count: int = 12,
) -> list[Path]:
    keyframes = motion.get("keyframes", [])
    if not keyframes:
        raise RuntimeError("Motion 没有 keyframes")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=480)
    cam = _camera()

    written: list[Path] = []
    try:
        for idx in sample_indices(len(keyframes), count):
            kf = keyframes[idx]
            image = _render_pose(model, data, renderer, cam, kf["pose"])
            path = out / f"robot_f{idx:05d}_t{float(kf['t']):.2f}.png"
            Image.fromarray(image).save(path)
            written.append(path)
    finally:
        renderer.close()
    return written


def render_pose_set(
    poses: list[tuple[str, dict[str, float]]],
    model_path: str | Path,
    out_dir: str | Path,
) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=480)
    cam = _camera()

    written: list[Path] = []
    try:
        for label, pose in poses:
            image = _render_pose(model, data, renderer, cam, pose)
            path = out / f"{label}.png"
            Image.fromarray(image).save(path)
            written.append(path)
    finally:
        renderer.close()
    return written


def stitch_comparison(human_paths: list[Path], robot_paths: list[Path], out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for idx, (human_path, robot_path) in enumerate(zip(human_paths, robot_paths)):
        human = cv2.imdecode(np.fromfile(str(human_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        robot = cv2.imdecode(np.fromfile(str(robot_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if human is None or robot is None:
            continue
        target_h = 480
        human = cv2.resize(human, (int(human.shape[1] * target_h / human.shape[0]), target_h))
        robot = cv2.resize(robot, (480, 480))
        combined = np.full((480, human.shape[1] + 480, 3), 255, dtype=np.uint8)
        combined[:, :human.shape[1]] = human
        combined[:, human.shape[1]:] = robot
        cv2.putText(combined, "human 2D pose", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
        cv2.putText(combined, "robot pose", (human.shape[1] + 16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
        path = out / f"compare_{idx:02d}.png"
        ok, encoded = cv2.imencode(".png", combined)
        if not ok:
            continue
        encoded.tofile(str(path))
        written.append(path)
    return written
