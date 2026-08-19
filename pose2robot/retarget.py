# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .constants import LM, ROBOT_JOINTS
from .robot import clip_pose, load_joint_limits


def _norm(value: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(value))
    if length < 1e-8:
        return np.zeros_like(value)
    return value / length


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    return math.acos(float(np.clip(np.dot(_norm(a), _norm(b)), -1.0, 1.0)))


def _body_frame(world: list[list[float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(world, dtype=float)
    left_shoulder = pts[LM["left_shoulder"]]
    right_shoulder = pts[LM["right_shoulder"]]
    left_hip = pts[LM["left_hip"]]
    right_hip = pts[LM["right_hip"]]
    shoulder_mid = (left_shoulder + right_shoulder) / 2.0
    hip_mid = (left_hip + right_hip) / 2.0

    x_right = _norm(right_shoulder - left_shoulder)
    y_up = _norm(shoulder_mid - hip_mid)
    z_forward = _norm(np.cross(x_right, y_up))
    if np.linalg.norm(z_forward) < 1e-8:
        z_forward = np.array([0.0, 0.0, 1.0])
    y_up = _norm(np.cross(z_forward, x_right))
    return shoulder_mid, hip_mid, x_right, y_up, z_forward


def _arm(world: list[list[float]], side: str, x_right: np.ndarray, y_up: np.ndarray, z_forward: np.ndarray) -> tuple[float, float, float]:
    pts = np.asarray(world, dtype=float)
    shoulder = pts[LM[f"{side}_shoulder"]]
    elbow = pts[LM[f"{side}_elbow"]]
    wrist = pts[LM[f"{side}_wrist"]]
    upper = _norm(elbow - shoulder)
    lower = _norm(wrist - elbow)

    up = float(np.dot(upper, y_up))
    forward = float(np.dot(upper, z_forward))
    pitch = -math.atan2(forward, -up)

    lateral_axis = -x_right if side == "left" else x_right
    lateral = float(np.dot(upper, lateral_axis))
    roll_mag = math.atan2(lateral, -up)
    roll = roll_mag if side == "left" else -roll_mag

    bend = _angle(upper, lower)
    elbow_value = 1.0 - bend / (math.pi / 2.0)
    return pitch, roll, elbow_value


def retarget_frame(frame: dict, limits: dict[str, tuple[float, float]]) -> dict[str, float]:
    world = frame["world"]
    pts = np.asarray(world, dtype=float)
    shoulder_mid, hip_mid, x_right, y_up, z_forward = _body_frame(world)
    torso = shoulder_mid - hip_mid

    nose = pts[LM["nose"]]
    head = nose - shoulder_mid
    head_yaw = math.atan2(float(np.dot(head, x_right)), abs(float(np.dot(head, z_forward))) + 1e-4)
    head_pitch = -math.atan2(float(np.dot(head, y_up)), abs(float(np.dot(head, z_forward))) + 1e-4) + 0.9

    waist_yaw = math.atan2(float(torso[0]), abs(float(torso[1])) + 1e-4)
    waist_pitch = math.atan2(float(torso[2]), abs(float(torso[1])) + 1e-4)

    left_pitch, left_roll, left_elbow = _arm(world, "left", x_right, y_up, z_forward)
    right_pitch, right_roll, right_elbow = _arm(world, "right", x_right, y_up, z_forward)

    raw = {
        "head_yaw_joint": head_yaw,
        "head_pitch_joint": head_pitch,
        "waist_yaw_joint": waist_yaw,
        "waist_pitch_joint": waist_pitch,
        "left_shoulder_pitch_joint": left_pitch,
        "left_shoulder_roll_joint": left_roll,
        "left_elbow_joint": left_elbow,
        "right_shoulder_pitch_joint": right_pitch,
        "right_shoulder_roll_joint": right_roll,
        "right_elbow_joint": right_elbow,
    }
    return clip_pose(raw, limits)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window < 3 or len(values) < window:
        return values
    if window % 2 == 0:
        window += 1
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def pose_to_joint_rows(pose_data: dict, model_path: str | Path) -> tuple[list[dict], dict[str, tuple[float, float]]]:
    limits = load_joint_limits(model_path)
    rows = []
    for frame in pose_data["landmarks"]:
        pose = retarget_frame(frame, limits)
        row = {"frame": frame.get("frame", len(rows)), "t": float(frame["t"])}
        row.update({name: pose[name] for name in ROBOT_JOINTS})
        rows.append(row)
    return rows, limits


def rows_to_motion(rows: list[dict], name: str, fps: float, smooth_window: int = 7) -> dict:
    if not rows:
        raise RuntimeError("没有可映射的关节帧")
    joint_arrays = {}
    for joint in ROBOT_JOINTS:
        values = np.asarray([row[joint] for row in rows], dtype=float)
        joint_arrays[joint] = moving_average(values, smooth_window)

    src_fps = 1.0 / max(1e-6, rows[1]["t"] - rows[0]["t"]) if len(rows) > 1 else fps
    step = max(1, int(round(src_fps / fps))) if fps > 0 else 1
    keyframes = []
    for idx in range(0, len(rows), step):
        keyframes.append({
            "t": float(rows[idx]["t"]),
            "pose": {joint: float(joint_arrays[joint][idx]) for joint in ROBOT_JOINTS},
        })
    return {
        "schema": "pose2robot.motion.v1",
        "name": name,
        "description": "Generated by pose2robot v2 clean pipeline",
        "keyframes": keyframes,
    }

