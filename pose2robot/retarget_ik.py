# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .constants import LM, ROBOT_JOINTS


def _norm(value: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(value))
    if length < 1e-8:
        return np.zeros_like(value)
    return value / length


@dataclass
class ArmTarget:
    upper: np.ndarray
    lower: np.ndarray


@dataclass
class LegTarget:
    thigh: np.ndarray
    shin: np.ndarray
    foot: np.ndarray


class UpperBodyIK:
    """MuJoCo FK 驱动的上半身数值 IK。

    目标不是解完整末端位姿,而是让机器人上臂/前臂方向匹配人体上臂/前臂方向。
    这样不会再依赖 shoulder_pitch/roll/yaw 名字的直觉语义。
    """

    def __init__(self, model_path: str | Path):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.limits = self._limits()
        self.qadr = self._qadr()

    def _limits(self) -> dict[str, tuple[float, float]]:
        limits = {}
        for jid in range(self.model.njnt):
            name = self.model.joint(jid).name
            limits[name] = (float(self.model.jnt_range[jid][0]), float(self.model.jnt_range[jid][1]))
        return limits

    def _qadr(self) -> dict[str, int]:
        qadr = {}
        for jid in range(self.model.njnt):
            name = self.model.joint(jid).name
            qadr[name] = int(self.model.joint(jid).qposadr[0])
        return qadr

    def _body_pos(self, name: str) -> np.ndarray:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise RuntimeError(f"URDF/MuJoCo body 不存在: {name}")
        return np.asarray(self.data.xpos[bid], dtype=float).copy()

    def _set_pose(self, pose: dict[str, float]) -> None:
        self.data.qpos[:] = self.model.qpos0
        for joint, value in pose.items():
            lo, hi = self.limits[joint]
            self.data.qpos[self.qadr[joint]] = max(lo, min(hi, float(value)))
        mujoco.mj_forward(self.model, self.data)

    def _arm_vectors(self, side: str, pose: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
        self._set_pose(pose)
        shoulder = self._body_pos(f"{side}_shoulder_pitch_link")
        elbow = self._body_pos(f"{side}_elbow_link")
        wrist = self._body_pos(f"{side}_wrist_roll_link")
        return _norm(elbow - shoulder), _norm(wrist - elbow)

    def _arm_error(self, side: str, pose: dict[str, float], target: ArmTarget) -> float:
        upper, lower = self._arm_vectors(side, pose)
        return float(np.sum((upper - target.upper) ** 2) + 0.8 * np.sum((lower - target.lower) ** 2))

    def _leg_vectors(self, side: str, pose: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._set_pose(pose)
        hip = self._body_pos(f"{side}_hip_yaw_link")
        knee = self._body_pos(f"{side}_knee_link")
        ankle = self._body_pos(f"{side}_ankle_roll_link")
        foot = self._body_pos(f"{side}_ankle_pitch_link")
        return _norm(knee - hip), _norm(ankle - knee), _norm(foot - ankle)

    def _leg_error(self, side: str, pose: dict[str, float], target: LegTarget) -> float:
        thigh, shin, foot = self._leg_vectors(side, pose)
        return float(
            np.sum((thigh - target.thigh) ** 2)
            + np.sum((shin - target.shin) ** 2)
            + 0.25 * np.sum((foot - target.foot) ** 2)
        )

    def solve_arm(self, side: str, target: ArmTarget, previous: dict[str, float] | None = None) -> dict[str, float]:
        prefix = f"{side}_"
        joints = [
            f"{prefix}shoulder_pitch_joint",
            f"{prefix}shoulder_roll_joint",
            f"{prefix}shoulder_yaw_joint",
            f"{prefix}elbow_joint",
        ]
        pose = {joint: 0.0 for joint in joints}
        if previous:
            pose.update({joint: float(previous.get(joint, 0.0)) for joint in joints})

        best_error = self._arm_error(side, pose, target)
        for step in (0.75, 0.35, 0.16, 0.08, 0.035, 0.015):
            improved = True
            while improved:
                improved = False
                for joint in joints:
                    lo, hi = self.limits[joint]
                    base = pose[joint]
                    for direction in (-1.0, 1.0):
                        candidate = dict(pose)
                        candidate[joint] = max(lo, min(hi, base + direction * step))
                        error = self._arm_error(side, candidate, target)
                        if error + 1e-9 < best_error:
                            pose = candidate
                            best_error = error
                            improved = True
        return pose

    def solve_leg(self, side: str, target: LegTarget, previous: dict[str, float] | None = None) -> dict[str, float]:
        prefix = f"{side}_"
        joints = [
            f"{prefix}hip_pitch_joint",
            f"{prefix}hip_roll_joint",
            f"{prefix}hip_yaw_joint",
            f"{prefix}knee_joint",
            f"{prefix}ankle_pitch_joint",
            f"{prefix}ankle_roll_joint",
        ]
        pose = {joint: 0.0 for joint in joints}
        if previous:
            pose.update({joint: float(previous.get(joint, 0.0)) for joint in joints})

        best_error = self._leg_error(side, pose, target)
        for step in (0.65, 0.3, 0.14, 0.065, 0.03):
            improved = True
            while improved:
                improved = False
                for joint in joints:
                    lo, hi = self.limits[joint]
                    base = pose[joint]
                    for direction in (-1.0, 1.0):
                        candidate = dict(pose)
                        candidate[joint] = max(lo, min(hi, base + direction * step))
                        error = self._leg_error(side, candidate, target)
                        if error + 1e-9 < best_error:
                            pose = candidate
                            best_error = error
                            improved = True
        return pose

    def solve_frame(self, frame: dict, previous: dict[str, float] | None = None) -> dict[str, float]:
        arm_targets = human_arm_targets(frame)
        leg_targets = human_leg_targets(frame)
        pose = {}
        pose.update(self.solve_leg("left", leg_targets["left"], previous))
        pose.update(self.solve_leg("right", leg_targets["right"], previous))
        pose.update(self.solve_arm("left", arm_targets["left"], previous))
        pose.update(self.solve_arm("right", arm_targets["right"], previous))
        pose.update(simple_head_waist(frame, self.limits))
        return {joint: float(pose.get(joint, 0.0)) for joint in ROBOT_JOINTS}


def _human_basis(world: list[list[float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(world, dtype=float)
    left_shoulder = pts[LM["left_shoulder"]]
    right_shoulder = pts[LM["right_shoulder"]]
    left_hip = pts[LM["left_hip"]]
    right_hip = pts[LM["right_hip"]]
    shoulder_mid = (left_shoulder + right_shoulder) / 2.0
    hip_mid = (left_hip + right_hip) / 2.0

    left = _norm(left_shoulder - right_shoulder)
    up = _norm(shoulder_mid - hip_mid)
    forward = _norm(np.cross(left, up))
    if np.linalg.norm(forward) < 1e-8:
        forward = np.array([0.0, 0.0, 1.0])
    up = _norm(np.cross(forward, left))
    return forward, left, up


def _to_robot_vector(vec: np.ndarray, basis: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    forward, left, up = basis
    return _norm(np.array([
        float(np.dot(vec, forward)),
        float(np.dot(vec, left)),
        float(np.dot(vec, up)),
    ]))


def human_arm_targets(frame: dict) -> dict[str, ArmTarget]:
    world = frame["world"]
    pts = np.asarray(world, dtype=float)
    basis = _human_basis(world)
    targets = {}
    for side in ("left", "right"):
        shoulder = pts[LM[f"{side}_shoulder"]]
        elbow = pts[LM[f"{side}_elbow"]]
        wrist = pts[LM[f"{side}_wrist"]]
        targets[side] = ArmTarget(
            upper=_to_robot_vector(elbow - shoulder, basis),
            lower=_to_robot_vector(wrist - elbow, basis),
        )
    return targets


def human_leg_targets(frame: dict) -> dict[str, LegTarget]:
    world = frame["world"]
    pts = np.asarray(world, dtype=float)
    basis = _human_basis(world)
    targets = {}
    for side in ("left", "right"):
        hip = pts[LM[f"{side}_hip"]]
        knee = pts[LM[f"{side}_knee"]]
        ankle = pts[LM[f"{side}_ankle"]]
        foot = pts[LM[f"{side}_foot_index"]]
        targets[side] = LegTarget(
            thigh=_to_robot_vector(knee - hip, basis),
            shin=_to_robot_vector(ankle - knee, basis),
            foot=_to_robot_vector(foot - ankle, basis),
        )
    return targets


def simple_head_waist(frame: dict, limits: dict[str, tuple[float, float]]) -> dict[str, float]:
    world = frame["world"]
    pts = np.asarray(world, dtype=float)
    forward, left, up = _human_basis(world)
    shoulder_mid = (pts[LM["left_shoulder"]] + pts[LM["right_shoulder"]]) / 2.0
    hip_mid = (pts[LM["left_hip"]] + pts[LM["right_hip"]]) / 2.0
    head = pts[LM["nose"]] - shoulder_mid
    torso = shoulder_mid - hip_mid

    raw = {
        "head_yaw_joint": float(np.arctan2(np.dot(head, left), abs(np.dot(head, forward)) + 1e-4)),
        "head_pitch_joint": float(-np.arctan2(np.dot(head, up), abs(np.dot(head, forward)) + 1e-4) + 0.9),
        "waist_yaw_joint": float(np.arctan2(torso[0], abs(torso[1]) + 1e-4)),
        "waist_pitch_joint": float(np.arctan2(torso[2], abs(torso[1]) + 1e-4)),
    }
    clipped = {}
    for joint, value in raw.items():
        lo, hi = limits[joint]
        clipped[joint] = max(lo, min(hi, value))
    return clipped


def pose_to_joint_rows_ik(pose_data: dict, model_path: str | Path) -> tuple[list[dict], dict[str, tuple[float, float]]]:
    solver = UpperBodyIK(model_path)
    previous = None
    rows = []
    for frame in pose_data["landmarks"]:
        pose = solver.solve_frame(frame, previous)
        previous = pose
        row = {"frame": frame.get("frame", len(rows)), "t": float(frame["t"])}
        row.update({name: pose[name] for name in ROBOT_JOINTS})
        rows.append(row)
    return rows, solver.limits
