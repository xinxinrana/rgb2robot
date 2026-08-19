r"""灵龙2.0 控制台 (PyQt5) — 替代 MuJoCo viewer 自带控制面板。

界面:
  - 30 个关节滑块(按身体部位分组)
  - 浮动基座高度
  - 播放/暂停/归零/保存/加载
  - 动作序列下拉 + ▶ 播放动作 + 进度条 (本次新增)
  - 4 个预置动作: 恭喜恭喜 / 爵士谢礼 / 累倒下 / 深蹲起立

为什么 PyQt5 而不是 PySide6: 本机 conda 环境装 PySide6 6.11 时
Qt6Core.dll 加载失败(Windows 缺少部分 VC++ 运行时 API),PyQt5 一次成功。

用法:
    C:\Users\xy\miniconda3\python.exe sim\panel.py
    或: run_panel.bat (双击)
"""
from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import mujoco
import mujoco.viewer

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QSlider, QPushButton, QDoubleSpinBox, QComboBox,
    QFileDialog, QMessageBox, QStatusBar, QSizePolicy, QProgressBar,
    QPlainTextEdit, QSplitter,
)

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "LingLong2.0" / "LingLong2.0.urdf"
LOG_FILE = Path(__file__).resolve().parent.parent / "panel.log"
SAVED_MOTIONS_DIR = Path(__file__).resolve().parent / "saved_motions"

JOINT_GROUPS: list[tuple[str, list[str]]] = [
    ("左腿", [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    ]),
    ("右腿", [
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    ]),
    ("腰部 + 头部", [
        "waist_yaw_joint", "waist_pitch_joint",
        "head_yaw_joint", "head_pitch_joint",
    ]),
    ("左臂", [
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    ]),
    ("右臂", [
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    ]),
]

# 老的挥手示例(连续循环)
PLAY_DEMO: dict[str, Callable[[float], float]] = {
    "right_shoulder_pitch_joint": lambda t: -0.9 + 0.4 * np.sin(t * 2.0),
    "right_shoulder_roll_joint":  lambda t: 0.5,
    "right_elbow_joint":          lambda t: 1.2,
    "left_shoulder_pitch_joint":  lambda t: -0.9 - 0.4 * np.sin(t * 2.0),
    "left_shoulder_roll_joint":   lambda t: -0.5,
    "left_elbow_joint":           lambda t: 1.2,
    "head_yaw_joint":             lambda t: 0.4 * np.sin(t * 0.7),
}

SLIDER_RES = 1000

# ---- T-pose / 自然站姿 模板(全身 30 关节) ----
# 注意:模型没有 free joint(无浮动基座),所以没有 base_x/y/z/q* 字段。
# T_POSE 是"自然站姿": 大臂贴身体 + 小臂自然下垂(elbow=+1.0;elbow=0 是前伸 90°)。
# 弧度制。左右肩 roll 镜像:左正右负,双臂水平展开。
T_POSE: dict[str, float] = {
    # 腿
    "left_hip_pitch_joint": 0.0, "left_hip_roll_joint": 0.0, "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.0, "left_ankle_pitch_joint": 0.0, "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": 0.0, "right_hip_roll_joint": 0.0, "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.0, "right_ankle_pitch_joint": 0.0, "right_ankle_roll_joint": 0.0,
    # 腰 + 头
    "waist_yaw_joint": 0.0, "waist_pitch_joint": 0.0,
    "head_yaw_joint": 0.0, "head_pitch_joint": 0.0,
    # 左臂:外展 90° + 小臂下垂
    "left_shoulder_pitch_joint": 0.0, "left_shoulder_roll_joint": 1.57,
    "left_shoulder_yaw_joint": 0.0, "left_elbow_joint": 1.0,
    "left_wrist_roll_joint": 0.0, "left_wrist_pitch_joint": 0.0, "left_wrist_yaw_joint": 0.0,
    # 右臂:外展 90°(镜像)+ 小臂下垂
    "right_shoulder_pitch_joint": 0.0, "right_shoulder_roll_joint": -1.57,
    "right_shoulder_yaw_joint": 0.0, "right_elbow_joint": 1.0,
    "right_wrist_roll_joint": 0.0, "right_wrist_pitch_joint": 0.0, "right_wrist_yaw_joint": 0.0,
}


def pose(**overrides) -> dict[str, float]:
    """生成一个完整 pose,以 T_POSE 为底,叠加 overrides。"""
    p = dict(T_POSE)
    p.update(overrides)
    return p


def quat_from_axis_angle(axis: tuple[float, float, float], angle_rad: float) -> tuple[float, float, float, float]:
    """axis 单位向量,angle 弧度 -> (qw, qx, qy, qz)。"""
    ax = np.array(axis, dtype=float)
    n = np.linalg.norm(ax)
    if n == 0:
        return (1.0, 0.0, 0.0, 0.0)
    ax = ax / n
    s = math.sin(angle_rad / 2.0)
    return (math.cos(angle_rad / 2.0), float(ax[0] * s), float(ax[1] * s), float(ax[2] * s))


# ---- 动作序列数据结构 ----
@dataclass
class Keyframe:
    t: float  # 从动作开始的秒数
    pose: dict[str, float]  # 完整 37 维 pose


@dataclass
class Motion:
    name: str
    description: str
    keyframes: list[Keyframe]

    @property
    def duration(self) -> float:
        return self.keyframes[-1].t if self.keyframes else 0.0

    def sample(self, t: float, default_pose: dict[str, float] | None = None) -> dict[str, float]:
        """在 t 时刻线性插值,clamp 到 [0, duration]。
        default_pose 用于填补关键帧里没写的字段(比如 base_z),避免落到 0/默认值。"""
        default = default_pose or {}
        if not self.keyframes:
            return {}
        if t <= self.keyframes[0].t:
            return dict(self.keyframes[0].pose)
        if t >= self.keyframes[-1].t:
            return dict(self.keyframes[-1].pose)
        for i in range(len(self.keyframes) - 1):
            k0, k1 = self.keyframes[i], self.keyframes[i + 1]
            if k0.t <= t <= k1.t:
                span = k1.t - k0.t
                alpha = 0.0 if span == 0 else (t - k0.t) / span
                keys = set(k0.pose) | set(k1.pose)
                return {n: k0.pose.get(n, default.get(n, 0.0)) * (1 - alpha)
                          + k1.pose.get(n, default.get(n, 0.0)) * alpha for n in keys}
        return dict(self.keyframes[-1].pose)


# ---- 已保存动作(从 SAVED_MOTIONS_DIR 加载,自动覆盖内置同名动作) ----
def load_all_saved_motions() -> dict[str, Motion]:
    """扫描 sim/saved_motions/ 下的 *.json,加载为 Motion 字典 {id: Motion}。
    启动时调用,自动覆盖 MOTIONS 里同名条目,这样用户的修改能跨会话保留。"""
    SAVED_MOTIONS_DIR.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Motion] = {}
    for f in sorted(SAVED_MOTIONS_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            motion = Motion(
                name=data["name"],
                description=data.get("description", ""),
                keyframes=[Keyframe(t=kf["t"], pose=kf["pose"]) for kf in data["keyframes"]],
            )
            saved[f.stem] = motion
        except Exception as e:
            print(f"[load] 跳过 {f.name}: {e}", file=sys.stderr)
    return saved

# 镜像规则: pitch / elbow / knee 镜像同号, roll / yaw 镜像反号
def _mirror_sign(joint_name):
    base = joint_name.rsplit("_joint", 1)[0]
    last = base.rsplit("_", 1)[-1]
    if last in ("pitch", "elbow", "knee"):
        return 1
    if last in ("roll", "yaw"):
        return -1
    return 1


def mirror_pose_dict(pose):
    """把一个 pose 的所有 left_* 映射成 right_*,应用镜像符号规则。

    返回的字典只包含 right_* 推导结果(以及非 left/right 的关节原样保留)。
    已存在的 right_* 输入会被跳过 —— 这函数的用途是"把左手侧的值复制
    /覆盖到右手侧",调用方拿到 result 后会把它 merge 回原 pose。
    """
    result = {}
    for name, val in pose.items():
        if name.startswith("left_"):
            right = "right_" + name[len("left_"):]
            result[right] = _mirror_sign(name) * val
        elif not name.startswith("right_"):
            # 非配对关节(头、腰 等)原样保留
            result[name] = val
    return result


def mirror_motion(motion, new_id, new_name):
    """把整个 motion 的所有关键帧做 L↔R 镜像,生成新 Motion。"""
    return Motion(
        name=new_name,
        description=f"镜像自「{motion.name}」",
        keyframes=[
            Keyframe(t=kf.t, pose=mirror_pose_dict(kf.pose))
            for kf in motion.keyframes
        ],
    )


# ---- 4 个预置动作 ----
MOTIONS: dict[str, Motion] = {
    "congratulations": Motion(
        name="恭喜恭喜",
        description="抬手胸前合十 → 弯腰 3 次(快节奏)→ 收势回 T-pose",
        keyframes=[
            Keyframe(0.0, T_POSE),
            Keyframe(0.5, pose(  # 抬到胸前,双手合十姿态(参考值: 左 pitch=-0.789, roll=-0.006, yaw=-0.997, elbow=-0.324;右臂镜像)
                left_shoulder_pitch_joint=-0.789, right_shoulder_pitch_joint=-0.789,
                left_shoulder_roll_joint=-0.006, right_shoulder_roll_joint=0.006,
                left_shoulder_yaw_joint=-0.997, right_shoulder_yaw_joint=0.997,
                left_elbow_joint=-0.324, right_elbow_joint=-0.324,
            )),
            # 鞠躬 1
            Keyframe(1.0, pose(  # 弯到最底
                left_shoulder_pitch_joint=-0.789, right_shoulder_pitch_joint=-0.789,
                left_shoulder_roll_joint=-0.006, right_shoulder_roll_joint=0.006,
                left_shoulder_yaw_joint=-0.997, right_shoulder_yaw_joint=0.997,
                left_elbow_joint=-0.324, right_elbow_joint=-0.324,
                waist_pitch_joint=0.4, head_pitch_joint=0.3,
            )),
            Keyframe(1.3, pose(  # 起立
                left_shoulder_pitch_joint=-0.789, right_shoulder_pitch_joint=-0.789,
                left_shoulder_roll_joint=-0.006, right_shoulder_roll_joint=0.006,
                left_shoulder_yaw_joint=-0.997, right_shoulder_yaw_joint=0.997,
                left_elbow_joint=-0.324, right_elbow_joint=-0.324,
            )),
            # 鞠躬 2
            Keyframe(1.8, pose(
                left_shoulder_pitch_joint=-0.789, right_shoulder_pitch_joint=-0.789,
                left_shoulder_roll_joint=-0.006, right_shoulder_roll_joint=0.006,
                left_shoulder_yaw_joint=-0.997, right_shoulder_yaw_joint=0.997,
                left_elbow_joint=-0.324, right_elbow_joint=-0.324,
                waist_pitch_joint=0.5, head_pitch_joint=0.3,
            )),
            Keyframe(2.1, pose(
                left_shoulder_pitch_joint=-0.789, right_shoulder_pitch_joint=-0.789,
                left_shoulder_roll_joint=-0.006, right_shoulder_roll_joint=0.006,
                left_shoulder_yaw_joint=-0.997, right_shoulder_yaw_joint=0.997,
                left_elbow_joint=-0.324, right_elbow_joint=-0.324,
            )),
            # 鞠躬 3(最重)
            Keyframe(2.6, pose(
                left_shoulder_pitch_joint=-0.789, right_shoulder_pitch_joint=-0.789,
                left_shoulder_roll_joint=-0.006, right_shoulder_roll_joint=0.006,
                left_shoulder_yaw_joint=-0.997, right_shoulder_yaw_joint=0.997,
                left_elbow_joint=-0.324, right_elbow_joint=-0.324,
                waist_pitch_joint=0.5, head_pitch_joint=0.3,
            )),
            Keyframe(3.2, pose(
                left_shoulder_pitch_joint=-0.789, right_shoulder_pitch_joint=-0.789,
                left_shoulder_roll_joint=-0.006, right_shoulder_roll_joint=0.006,
                left_shoulder_yaw_joint=-0.997, right_shoulder_yaw_joint=0.997,
                left_elbow_joint=-0.324, right_elbow_joint=-0.324,
            )),
            # 收势
            Keyframe(4.0, T_POSE),
        ],
    ),
    "jazz_bow": Motion(
        name="爵士谢礼",
        description="右手抚胸、左臂外展、鞠躬、起立",
        keyframes=[
            Keyframe(0.0, T_POSE),
            Keyframe(0.6, pose(
                # 右手抚胸
                right_shoulder_pitch_joint=-0.4, right_shoulder_roll_joint=0.08,
                right_shoulder_yaw_joint=0.0, right_elbow_joint=1.2,
                # 左臂轻外展
                left_shoulder_roll_joint=0.5,
            )),
            Keyframe(1.4, pose(
                # 右手外展(爵士展姿)
                right_shoulder_pitch_joint=0.0, right_shoulder_roll_joint=-0.7,
                right_shoulder_yaw_joint=0.0, right_elbow_joint=0.3,
                right_wrist_pitch_joint=-0.3,
                # 左手在身侧
                left_shoulder_roll_joint=0.4, left_elbow_joint=0.2,
            )),
            Keyframe(2.1, pose(
                right_shoulder_pitch_joint=0.0, right_shoulder_roll_joint=-0.7,
                right_elbow_joint=0.3, right_wrist_pitch_joint=-0.3,
                left_shoulder_roll_joint=0.4, left_elbow_joint=0.2,
                waist_pitch_joint=0.4, head_pitch_joint=0.2,  # 弯腰+低头
            )),
            Keyframe(2.8, pose(
                right_shoulder_pitch_joint=0.0, right_shoulder_roll_joint=-0.7,
                right_elbow_joint=0.3, right_wrist_pitch_joint=-0.3,
                left_shoulder_roll_joint=0.4, left_elbow_joint=0.2,
                waist_pitch_joint=0.0, head_pitch_joint=0.0,
            )),
            Keyframe(3.5, T_POSE),
        ],
    ),
    "collapse": Motion(
        name="累倒下",
        description="低头 → 肩沉 → 屈膝 → 深蹲+驼背(模型无 free joint,不能真倒地)",
        keyframes=[
            Keyframe(0.0, T_POSE),
            Keyframe(0.6, pose(  # 微微低头 + 肩沉
                head_pitch_joint=0.2,
                left_shoulder_pitch_joint=0.2, right_shoulder_pitch_joint=0.2,  # 大臂略后摆=肩沉
                left_elbow_joint=0.2, right_elbow_joint=0.2,
            )),
            Keyframe(1.4, pose(  # 低头 + 肩更沉 + 微屈膝
                head_pitch_joint=0.4,
                left_shoulder_pitch_joint=0.5, right_shoulder_pitch_joint=0.5,
                left_elbow_joint=0.6, right_elbow_joint=0.6,
                left_knee_joint=0.6, right_knee_joint=0.6,
            )),
            Keyframe(2.4, pose(  # 深蹲 + 肩前合
                head_pitch_joint=0.5,
                left_shoulder_pitch_joint=0.7, right_shoulder_pitch_joint=0.7,  # 大臂后摆=肩驼
                left_shoulder_roll_joint=0.4, right_shoulder_roll_joint=-0.4,  # 略外展
                left_elbow_joint=0.9, right_elbow_joint=0.9,
                left_knee_joint=1.5, right_knee_joint=1.5,  # 深蹲
            )),
            Keyframe(3.5, pose(  # 极限:深蹲到最低 + 头垂到最下 + 肩完全前合
                head_pitch_joint=0.6,
                left_shoulder_pitch_joint=0.9, right_shoulder_pitch_joint=0.9,
                left_shoulder_roll_joint=0.6, right_shoulder_roll_joint=-0.6,
                left_elbow_joint=1.2, right_elbow_joint=1.2,
                left_knee_joint=1.8, right_knee_joint=1.8,  # 接近限位上限
            )),
        ],
    ),
    "squat": Motion(
        name="深蹲起立",
        description="双臂前伸平衡 → 蹲到底 → 起立回到 T-pose",
        keyframes=[
            Keyframe(0.0, T_POSE),
            Keyframe(0.6, pose(  # 准备:双肩前伸 + 肘微弯 + 浅蹲
                left_shoulder_pitch_joint=-0.8, right_shoulder_pitch_joint=-0.8,
                left_shoulder_roll_joint=0.3, right_shoulder_roll_joint=-0.3,
                left_elbow_joint=0.3, right_elbow_joint=0.3,
                left_knee_joint=0.3, right_knee_joint=0.3,
            )),
            Keyframe(1.2, pose(  # 蹲底:knee=1.5 + 髋微屈 + 腰微前倾
                left_shoulder_pitch_joint=-0.5, right_shoulder_pitch_joint=-0.5,
                left_elbow_joint=0.6, right_elbow_joint=0.6,
                left_knee_joint=1.5, right_knee_joint=1.5,
                left_hip_pitch_joint=-0.3, right_hip_pitch_joint=-0.3,
                waist_pitch_joint=0.15, head_pitch_joint=0.1,
            )),
            Keyframe(1.8, pose(  # 起立回到准备姿
                left_shoulder_pitch_joint=-0.8, right_shoulder_pitch_joint=-0.8,
                left_shoulder_roll_joint=0.3, right_shoulder_roll_joint=-0.3,
                left_elbow_joint=0.3, right_elbow_joint=0.3,
                left_knee_joint=0.3, right_knee_joint=0.3,
            )),
            Keyframe(2.4, T_POSE),
        ],
    ),
    "one_take": Motion(
        name="一镜到底",
        description="60s 独角戏: 候场→惊吓→挥手→鞠躬→流行舞→跳起→太极→冲拳→独唱→谢幕",
        keyframes=[
            # === 第一幕: 候场与开场 (0-18s) ===
            Keyframe(0.0, T_POSE),  # 静候
            Keyframe(2.5, pose(  # 惊吓: 肩猛抬, 头微缩
                head_pitch_joint=0.1,
                left_shoulder_pitch_joint=0.5, right_shoulder_pitch_joint=0.5,
                left_shoulder_roll_joint=2.0, right_shoulder_roll_joint=-2.0,
                left_elbow_joint=0.8, right_elbow_joint=0.8,
            )),
            Keyframe(3.5, pose(  # 恢复
                left_shoulder_pitch_joint=0.0, right_shoulder_pitch_joint=0.0,
                left_shoulder_roll_joint=1.57, right_shoulder_roll_joint=-1.57,
                left_elbow_joint=0.0, right_elbow_joint=0.0,
                head_pitch_joint=0.0,
            )),
            Keyframe(4.5, pose(  # 看衣领
                head_pitch_joint=0.4, head_yaw_joint=0.2,
                left_shoulder_pitch_joint=0.2, right_shoulder_pitch_joint=0.2,
                left_shoulder_roll_joint=0.8, right_shoulder_roll_joint=-0.8,
                left_elbow_joint=1.0, right_elbow_joint=1.0,
            )),
            Keyframe(5.5, pose(  # 抬头
                head_pitch_joint=0.0, head_yaw_joint=0.0,
                left_shoulder_roll_joint=1.57, right_shoulder_roll_joint=-1.57,
            )),
            Keyframe(7.0, pose(  # 挥手起
                right_shoulder_pitch_joint=-1.0, right_shoulder_roll_joint=-0.5,
                right_elbow_joint=0.8, head_yaw_joint=0.2,
            )),
            Keyframe(8.0, pose(  # 挥手摆过
                right_shoulder_pitch_joint=-0.8, right_shoulder_roll_joint=0.0,
                right_shoulder_yaw_joint=-0.5, right_elbow_joint=0.6,
                head_yaw_joint=0.0,
            )),
            Keyframe(9.0, pose(  # 放下
                right_shoulder_pitch_joint=0.0, right_shoulder_roll_joint=-1.0,
                right_elbow_joint=0.0,
            )),
            Keyframe(11.0, pose(  # 鞠躬
                waist_pitch_joint=0.5, head_pitch_joint=0.3,
                left_shoulder_pitch_joint=0.3, right_shoulder_pitch_joint=0.3,
                left_shoulder_roll_joint=0.5, right_shoulder_roll_joint=-0.5,
                left_elbow_joint=0.5, right_elbow_joint=0.5,
            )),
            Keyframe(13.0, pose(  # 起身 + 单手指
                waist_pitch_joint=0.0, head_pitch_joint=-0.1,
                right_shoulder_pitch_joint=-1.0, right_shoulder_roll_joint=-0.3,
                right_elbow_joint=0.3,
            )),
            Keyframe(15.0, pose(  # 双手展开介绍
                left_shoulder_roll_joint=2.0, right_shoulder_roll_joint=-2.0,
                left_shoulder_pitch_joint=-0.3, right_shoulder_pitch_joint=-0.3,
                left_elbow_joint=0.3, right_elbow_joint=0.3,
            )),
            Keyframe(16.5, pose(  # 准备跳舞
                left_hip_yaw_joint=0.2, right_hip_yaw_joint=-0.2,
                left_knee_joint=0.3, right_knee_joint=0.3,
                left_hip_pitch_joint=-0.15, right_hip_pitch_joint=-0.15,
                left_shoulder_roll_joint=1.0, right_shoulder_roll_joint=-1.0,
                left_elbow_joint=0.8, right_elbow_joint=0.8,
                waist_pitch_joint=0.05,
            )),
            # === 第二幕: 表演 (18-42s) ===
            Keyframe(18.5, pose(  # 摆胯左
                left_hip_yaw_joint=-0.3, right_hip_yaw_joint=0.3,
                waist_yaw_joint=-0.3,
                left_shoulder_roll_joint=1.2, right_shoulder_roll_joint=-1.2,
                left_elbow_joint=0.8, right_elbow_joint=0.8,
            )),
            Keyframe(20.5, pose(  # 摆胯右
                left_hip_yaw_joint=0.3, right_hip_yaw_joint=-0.3,
                waist_yaw_joint=0.3,
                left_shoulder_roll_joint=1.2, right_shoulder_roll_joint=-1.2,
                left_elbow_joint=0.8, right_elbow_joint=0.8,
            )),
            Keyframe(22.5, pose(  # 双臂举高过头
                left_shoulder_roll_joint=2.8, right_shoulder_roll_joint=-2.8,  # roll 接近上限=举高
                left_shoulder_pitch_joint=0.0, right_shoulder_pitch_joint=0.0,
                left_elbow_joint=0.0, right_elbow_joint=0.0,
            )),
            Keyframe(24.5, pose(  # 拍大腿
                left_shoulder_pitch_joint=0.2, right_shoulder_pitch_joint=0.2,
                left_shoulder_roll_joint=0.3, right_shoulder_roll_joint=-0.3,
                left_elbow_joint=1.4, right_elbow_joint=1.4,
                waist_pitch_joint=0.15, head_pitch_joint=0.15,
            )),
            Keyframe(26.5, pose(  # 拍胸
                left_shoulder_pitch_joint=0.5, right_shoulder_pitch_joint=0.5,
                left_shoulder_roll_joint=0.5, right_shoulder_roll_joint=-0.5,
                left_elbow_joint=1.4, right_elbow_joint=1.4,
            )),
            Keyframe(28.0, pose(  # 预蹲(准备"起跳"姿态) — 模型没 free joint,改用深蹲代替
                left_knee_joint=0.8, right_knee_joint=0.8,
                left_hip_pitch_joint=-0.3, right_hip_pitch_joint=-0.3,
                left_shoulder_pitch_joint=-1.0, right_shoulder_pitch_joint=-1.0,  # 大臂前推
                left_shoulder_roll_joint=0.78, right_shoulder_roll_joint=-0.78,
                left_elbow_joint=0.3, right_elbow_joint=0.3,
            )),
            Keyframe(29.0, pose(  # 跳起峰值(模拟:深蹲到最底+举高过头) — 替代"假跳起"
                left_knee_joint=1.6, right_knee_joint=1.6,  # 深蹲
                left_hip_pitch_joint=-0.4, right_hip_pitch_joint=-0.4,
                left_shoulder_roll_joint=2.8, right_shoulder_roll_joint=-2.8,  # 举高过头
                left_shoulder_pitch_joint=0.0, right_shoulder_pitch_joint=0.0,
                left_elbow_joint=0.0, right_elbow_joint=0.0,
            )),
            Keyframe(30.0, pose(  # 落地缓冲:起立回半蹲
                left_knee_joint=0.4, right_knee_joint=0.4,
                left_hip_pitch_joint=-0.2, right_hip_pitch_joint=-0.2,
                left_shoulder_pitch_joint=-0.5, right_shoulder_pitch_joint=-0.5,
                left_shoulder_roll_joint=0.78, right_shoulder_roll_joint=-0.78,
                left_elbow_joint=0.5, right_elbow_joint=0.5,
            )),
            Keyframe(31.5, pose(  # 喘息(双手叉腰)
                left_shoulder_pitch_joint=0.3, right_shoulder_pitch_joint=0.3,
                left_shoulder_roll_joint=0.5, right_shoulder_roll_joint=-0.5,
                left_elbow_joint=1.4, right_elbow_joint=1.4,
                waist_pitch_joint=0.1, head_pitch_joint=0.1,
            )),
            Keyframe(35.0, pose(  # 太极起势(左脚前迈)
                left_hip_yaw_joint=0.4, right_hip_yaw_joint=0.2,
                left_knee_joint=0.6, right_knee_joint=0.2,
                left_hip_pitch_joint=-0.3, right_hip_pitch_joint=-0.1,
                left_shoulder_pitch_joint=-0.8, right_shoulder_pitch_joint=-0.5,
                left_shoulder_roll_joint=0.5, right_shoulder_roll_joint=-0.5,
                left_elbow_joint=0.6, right_elbow_joint=0.8,
                waist_yaw_joint=-0.2,
            )),
            Keyframe(37.0, pose(  # 推掌
                right_shoulder_pitch_joint=-1.0, right_shoulder_roll_joint=-0.5,
                right_elbow_joint=0.0,
                left_shoulder_pitch_joint=-0.5, left_shoulder_roll_joint=0.5,
                left_elbow_joint=1.0,
                waist_pitch_joint=0.1,
            )),
            Keyframe(38.5, pose(  # 收掌
                right_shoulder_pitch_joint=-0.3, right_shoulder_roll_joint=-0.3,
                right_elbow_joint=1.2,
                left_shoulder_pitch_joint=-0.5, left_shoulder_roll_joint=0.5,
                left_elbow_joint=1.0,
                waist_pitch_joint=0.05,
            )),
            Keyframe(40.5, pose(  # 弓步冲拳
                left_hip_pitch_joint=-0.5, right_hip_pitch_joint=-0.1,
                left_knee_joint=1.0, right_knee_joint=0.2,
                right_shoulder_pitch_joint=-0.8, right_shoulder_roll_joint=-0.5,
                right_elbow_joint=0.0,
                left_shoulder_pitch_joint=0.2, left_shoulder_roll_joint=0.5,
                left_elbow_joint=1.0,
                waist_yaw_joint=-0.3, head_yaw_joint=-0.2,
            )),
            Keyframe(42.0, T_POSE),  # 收势
            # === 第三幕: 深情独唱 + 谢幕 (42-60s) ===
            Keyframe(44.0, pose(  # 拿话筒
                right_shoulder_pitch_joint=-0.5, right_shoulder_roll_joint=-0.2,
                right_elbow_joint=1.4,
                left_shoulder_roll_joint=1.0,
                head_yaw_joint=-0.3,
            )),
            Keyframe(47.0, pose(  # 深情前倾
                right_shoulder_pitch_joint=-0.4, right_shoulder_roll_joint=-0.2,
                right_elbow_joint=1.3,
                left_shoulder_pitch_joint=-0.3, left_shoulder_roll_joint=0.5,
                left_elbow_joint=0.8,
                waist_pitch_joint=0.2, head_pitch_joint=0.15,
            )),
            Keyframe(49.0, pose(  # 高音后仰
                right_shoulder_pitch_joint=-1.0, right_shoulder_roll_joint=-0.3,
                right_elbow_joint=1.0,
                left_shoulder_pitch_joint=-1.0, left_shoulder_roll_joint=0.3,
                left_elbow_joint=0.5,
                waist_pitch_joint=-0.2, head_pitch_joint=-0.3,
            )),
            Keyframe(51.0, pose(  # 收麦
                right_shoulder_pitch_joint=0.0, right_shoulder_roll_joint=-1.0,
                right_elbow_joint=0.0,
                left_shoulder_roll_joint=1.57, left_elbow_joint=0.0,
                waist_pitch_joint=0.0, head_pitch_joint=0.0,
            )),
            Keyframe(53.0, pose(  # 站直准备谢幕
                left_shoulder_roll_joint=1.57, right_shoulder_roll_joint=-1.57,
                left_elbow_joint=0.0, right_elbow_joint=0.0,
            )),
            Keyframe(55.0, pose(  # 深鞠躬
                waist_pitch_joint=0.5, head_pitch_joint=0.3,
                left_shoulder_pitch_joint=0.3, right_shoulder_pitch_joint=0.3,
                left_shoulder_roll_joint=0.3, right_shoulder_roll_joint=-0.3,
                left_elbow_joint=1.0, right_elbow_joint=1.0,
                left_hip_pitch_joint=-0.2, right_hip_pitch_joint=-0.2,
            )),
            Keyframe(57.0, pose(  # 环视左
                head_yaw_joint=0.3,
            )),
            Keyframe(58.0, pose(  # 环视右
                head_yaw_joint=-0.3,
            )),
            Keyframe(59.5, pose(  # 挥手告别
                right_shoulder_pitch_joint=-0.8, right_shoulder_roll_joint=-0.3,
                right_elbow_joint=0.8,
                head_yaw_joint=0.0,
                left_shoulder_roll_joint=1.57, left_elbow_joint=0.0,
            )),
        ],
    ),
}


def short_label(name: str) -> str:
    n = name.replace("_joint", "")
    for prefix, repl in (("left_", "L_"), ("right_", "R_")):
        if n.startswith(prefix):
            return repl + n[len(prefix):]
    return n


class Controller:
    MODE_MANUAL = "manual"
    MODE_DEMO = "demo"
    MODE_MOTION = "motion"

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.joint_info: dict[str, tuple[int, float, float]] = {}
        for i in range(model.njnt):
            name = model.joint(i).name
            if not name:
                continue
            adr = model.joint(i).qposadr[0]
            lo, hi = float(model.jnt_range[i][0]), float(model.jnt_range[i][1])
            self.joint_info[name] = (adr, lo, hi)
        self.sliders: dict[str, tuple[QSlider, QLabel]] = {}
        self.mode_label: QLabel | None = None
        self.t_now_label: QLabel | None = None
        self.status_label: QLabel | None = None
        self.motion_combo: QComboBox | None = None
        self.motion_progress: QProgressBar | None = None
        self.log_view: QPlainTextEdit | None = None
        self._mode = self.MODE_MANUAL
        self._demo_t0 = 0.0
        self._motion_id: str | None = None
        self._motion_t0 = 0.0
        # 录制
        self._is_recording = False
        self._record_t0 = 0.0
        self._record_buffer = []
        # 时间轴滑块(由 build_ui 注入)
        self.t_scrub_slider = None
        self.record_btn = None
        self.record_status_lbl = None

        # 启动时清空 log 文件
        try:
            LOG_FILE.write_text("", encoding="utf-8")
        except OSError:
            pass

    def log(self, msg: str, level: str = "info") -> None:
        """追加一行到日志面板和 panel.log。level: info/ok/warn/error。"""
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] [{level.upper():<5s}] {msg}"
        if self.log_view is not None:
            self.log_view.appendPlainText(line)
            sb = self.log_view.verticalScrollBar()
            sb.setValue(sb.maximum())
        try:
            with LOG_FILE.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    @staticmethod
    def slider_to_val(s: int, lo: float, hi: float) -> float:
        return lo + (hi - lo) * s / SLIDER_RES

    @staticmethod
    def val_to_slider(v: float, lo: float, hi: float) -> int:
        if hi == lo:
            return 0
        return max(0, min(SLIDER_RES, int(round((v - lo) / (hi - lo) * SLIDER_RES))))

    def apply_manual(self) -> None:
        for name, (slider, _) in self.sliders.items():
            adr, lo, hi = self.joint_info[name]
            self.data.qpos[adr] = self.slider_to_val(slider.value(), lo, hi)

    def apply_demo(self, t: float) -> None:
        for name, fn in PLAY_DEMO.items():
            adr, lo, hi = self.joint_info[name]
            val = float(fn(t))
            clamped = max(lo, min(hi, val))
            self.data.qpos[adr] = clamped
            if name in self.sliders:
                slider, lbl = self.sliders[name]
                slider.blockSignals(True)
                slider.setValue(self.val_to_slider(clamped, lo, hi))
                slider.blockSignals(False)
                lbl.setText(f"{clamped:+.3f}")

    def apply_motion(self, t: float, motion: Motion) -> bool:
        """返回 False 表示动作已结束(应切回 manual)。"""
        if t >= motion.duration:
            self.set_mode(self.MODE_MANUAL)
            self.set_status(f"动作「{motion.name}」已结束 — 点播放重看,或选别的动作")
            self.log(f"动作「{motion.name}」播放完成 (t={motion.duration:.2f}s)", "ok")
            if self.motion_progress is not None:
                self.motion_progress.setValue(self.motion_progress.maximum())
            return False
        sample = motion.sample(t, {})
        for name, val in sample.items():
            if name not in self.joint_info:
                # 模型没有 free joint,理论上不会有 base_* 字段;留着防御性 warn
                self.log(f"  未知字段: {name}={val:.3f}(已忽略)", "warn")
                continue
            adr, lo, hi = self.joint_info[name]
            clamped = max(lo, min(hi, val))
            self.data.qpos[adr] = clamped
            if name in self.sliders:
                slider, lbl = self.sliders[name]
                slider.blockSignals(True)
                slider.setValue(self.val_to_slider(clamped, lo, hi))
                slider.blockSignals(False)
                lbl.setText(f"{clamped:+.3f}")
        return True

    def reset_zero(self) -> None:
        for slider, _ in self.sliders.values():
            slider.blockSignals(True)
            slider.setValue(0)
            slider.blockSignals(False)
        self.data.qpos[:] = self.model.qpos0
        mujoco.mj_forward(self.model, self.data)
        self._refresh_labels()
        self.set_mode(self.MODE_MANUAL)
        self._motion_id = None
        if self.motion_progress is not None:
            self.motion_progress.setValue(0)
            self.motion_progress.setFormat("— / — s")
        self.set_status("已归零 (model.qpos0)")
        self.log("⟲ 归零 → model.qpos0", "info")

    def _refresh_labels(self) -> None:
        for name, (slider, lbl) in self.sliders.items():
            _, lo, hi = self.joint_info[name]
            lbl.setText(f"{self.slider_to_val(slider.value(), lo, hi):+.3f}")

    def set_mode(self, mode: str) -> None:
        self._mode = mode
        if self.mode_label is not None:
            label = {"manual": "手动", "demo": "示例挥手", "motion": "动作序列"}[mode]
            color = {"manual": "#1565c0", "demo": "#ef6c00", "motion": "#c62828"}[mode]
            self.mode_label.setText(label)
            self.mode_label.setStyleSheet(f"color: {color}; font-weight: bold;")

    def set_status(self, msg: str) -> None:
        if self.status_label is not None:
            self.status_label.setText(msg)
        self.log(msg, "info")

    def start_demo(self) -> None:
        self._motion_id = None
        self._demo_t0 = time.time()
        self.set_mode(self.MODE_DEMO)
        if self.motion_progress is not None:
            self.motion_progress.setValue(0)
            self.motion_progress.setFormat("示例挥手(连续)")
        self.set_status("播放示例挥手 — 连续循环,按 ⏹停止")
        self.log("开始播放示例挥手(连续循环)", "info")

    def start_motion(self, motion_id: str) -> None:
        if motion_id not in MOTIONS:
            self.log(f"未知动作 ID: {motion_id}", "warn")
            return
        self._motion_id = motion_id
        self._motion_t0 = time.time()
        self.set_mode(self.MODE_MOTION)
        m = MOTIONS[motion_id]
        if self.motion_progress is not None:
            self.motion_progress.setRange(0, max(1, int(m.duration * 100)))
            self.motion_progress.setValue(0)
            self.motion_progress.setFormat(f"0.00 / {m.duration:.2f} s")
        self.set_status(f"播放动作「{m.name}」 — {m.description}")
        self.log(f"▶ 开始播放「{m.name}」({m.duration:.1f}s, {len(m.keyframes)} 关键帧)", "info")

    def stop_play(self) -> None:
        self._motion_id = None
        self.set_mode(self.MODE_MANUAL)
        if self.motion_progress is not None:
            self.motion_progress.setValue(0)
            self.motion_progress.setFormat("— / — s")
        self.set_status("已停止 — 拖动滑块改 pose")
        self.log("⏹ 停止播放", "info")

    def save_pose(self, path: Path) -> None:
        snap: dict[str, float] = {}
        for name, (slider, _) in self.sliders.items():
            _, lo, hi = self.joint_info[name]
            snap[name] = self.slider_to_val(slider.value(), lo, hi)
        path.write_text(json.dumps(snap, indent=2, ensure_ascii=False), encoding="utf-8")
        self.set_status(f"已保存 -> {path.name}")
        self.log(f"💾 保存 pose -> {path.name} ({len(snap)} 字段)", "info")

    def load_pose(self, path: Path) -> None:
        snap = json.loads(path.read_text(encoding="utf-8"))
        for name, val in snap.items():
            if name in self.sliders:
                _, lo, hi = self.joint_info[name]
                slider, _ = self.sliders[name]
                slider.blockSignals(True)
                slider.setValue(self.val_to_slider(float(val), lo, hi))
                slider.blockSignals(False)
        self.apply_manual()
        mujoco.mj_forward(self.model, self.data)
        self._refresh_labels()
        self.set_status(f"已加载 <- {path.name}")
        self.log(f"📂 加载 pose <- {path.name} ({len(snap)} 字段)", "info")

    def save_motion_to(self, motion: Motion, path: Path) -> None:
        """把整个动作(含全部关键帧)存到 JSON。"""
        data = {
            "name": motion.name,
            "description": motion.description,
            "keyframes": [{"t": kf.t, "pose": dict(kf.pose)} for kf in motion.keyframes],
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        self.log(f"💾 保存动作「{motion.name}」 -> {path.name} ({len(motion.keyframes)} 关键帧)", "ok")

    def load_motion_from(self, path: Path) -> tuple[str, Motion]:
        """从 JSON 读一个动作,返回 (id, Motion)。id 默认用文件名(无 .json 后缀)。"""
        data = json.loads(path.read_text(encoding="utf-8"))
        motion = Motion(
            name=data["name"],
            description=data.get("description", ""),
            keyframes=[Keyframe(t=kf["t"], pose=dict(kf["pose"])) for kf in data["keyframes"]],
        )
        return path.stem, motion

    def refresh_motion_combo(self) -> None:
        """按当前 MOTIONS 重新填充下拉框。已保存到磁盘的动作前加 ★ 标记。"""
        if self.motion_combo is None:
            return
        cur = self.motion_combo.currentData()
        self.motion_combo.clear()
        for mid, m in MOTIONS.items():
            mark = "★" if (SAVED_MOTIONS_DIR / f"{mid}.json").exists() else " "
            self.motion_combo.addItem(f"{mark} {m.name}", mid)
        if cur:
            idx = self.motion_combo.findData(cur)
            if idx >= 0:
                self.motion_combo.setCurrentIndex(idx)


    # ---- 镜像 ----
    def mirror_current_pose(self) -> None:
        """把当前 30 个 slider 的值做 L<->R 镜像,写回。"""
        pose = {}
        for name, (slider, _) in self.sliders.items():
            _, lo, hi = self.joint_info[name]
            pose[name] = self.slider_to_val(slider.value(), lo, hi)
        mirrored = mirror_pose_dict(pose)
        for name, val in mirrored.items():
            if name in self.sliders:
                _, lo, hi = self.joint_info[name]
                slider, lbl = self.sliders[name]
                slider.blockSignals(True)
                slider.setValue(self.val_to_slider(val, lo, hi))
                slider.blockSignals(False)
                lbl.setText(f"{val:+.3f}")
        self.apply_manual()
        mujoco.mj_forward(self.model, self.data)
        self.log("🪞 已镜像当前 pose (L<->R)", "ok")

    def mirror_current_motion(self) -> None:
        """把下拉框选中的 motion 做 L<->R 镜像,生成新 motion 加入下拉框。"""
        if self.motion_combo is None:
            return
        mid = self.motion_combo.currentData()
        if not mid or mid not in MOTIONS:
            self.log("未选中动作,无法镜像", "warn")
            return
        src_motion = MOTIONS[mid]
        new_id = f"{mid}_mirror"
        suffix = 1
        while new_id in MOTIONS:
            suffix += 1
            new_id = f"{mid}_mirror{suffix}"
        new_name = f"{src_motion.name}·镜像"
        MOTIONS[new_id] = mirror_motion(src_motion, new_id, new_name)
        self.refresh_motion_combo()
        idx = self.motion_combo.findData(new_id)
        if idx >= 0:
            self.motion_combo.setCurrentIndex(idx)
        self.log(f"🪞 已镜像动作「{src_motion.name}」->「{new_name}」({new_id})", "ok")

    # ---- 录制 ----
    def start_recording(self) -> None:
        self._is_recording = True
        self._record_t0 = time.time()
        self._record_buffer = []
        self.set_mode(self.MODE_MANUAL)
        if self.record_btn is not None:
            self.record_btn.setText("⏹ 停止录制")
            self.record_btn.setStyleSheet("background-color: #c62828; color: white; font-weight: bold;")
        if self.record_status_lbl is not None:
            self.record_status_lbl.setText("● 录制中 0 帧")
        self.log("⏺ 开始录制 — 手动拖 sliders,完事点停止", "ok")

    def stop_recording(self) -> None:
        if not self._is_recording:
            return
        self._is_recording = False
        if self.record_btn is not None:
            self.record_btn.setText("⏺ 开始录制")
            self.record_btn.setStyleSheet("")
        n = len(self._record_buffer)
        if n < 2:
            if self.record_status_lbl is not None:
                self.record_status_lbl.setText("")
            self.log(f"录制过短({n} 帧),已丢弃", "warn")
            self._record_buffer = []
            return
        from PyQt5.QtWidgets import QInputDialog
        default_name = f"录制 {time.strftime('%H:%M:%S')}"
        name, ok = QInputDialog.getText(None, "保存录制", "动作名:", text=default_name)
        if not ok or not name.strip():
            self.log("已取消保存", "info")
            self._record_buffer = []
            if self.record_status_lbl is not None:
                self.record_status_lbl.setText("")
            return
        name = name.strip()
        new_id = f"recorded_{int(time.time())}"
        kfs = sorted(self._record_buffer, key=lambda x: x[0])
        t0 = kfs[0][0]
        kfs = [Keyframe(t=t - t0, pose=dict(p)) for t, p in kfs]
        motion = Motion(
            name=name,
            description=f"录制于 {time.strftime('%Y-%m-%d %H:%M:%S')}, {n} 帧, {kfs[-1].t:.2f}s",
            keyframes=kfs,
        )
        MOTIONS[new_id] = motion
        self.refresh_motion_combo()
        if self.motion_combo is not None:
            self.motion_combo.setCurrentIndex(self.motion_combo.findData(new_id))
        if self.record_status_lbl is not None:
            self.record_status_lbl.setText("")
        self._record_buffer = []
        try:
            self.save_motion_to(motion, SAVED_MOTIONS_DIR / f"{new_id}.json")
        except Exception as e:
            self.log(f"自动保存失败: {e}", "warn")
        self.log(f"⏹ 录制完成 ->「{name}」({n} 帧, {kfs[-1].t:.2f}s)", "ok")

    def on_tick_recording(self) -> None:
        """tick() 调用:如果正在录制,采集当前 pose(只在变化时记一帧)。"""
        if not self._is_recording or self._mode != self.MODE_MANUAL:
            return
        t = time.time() - self._record_t0
        pose = {}
        for name, (slider, _) in self.sliders.items():
            _, lo, hi = self.joint_info[name]
            pose[name] = self.slider_to_val(slider.value(), lo, hi)
        if not self._record_buffer or self._record_buffer[-1][1] != pose:
            self._record_buffer.append((t, pose))
            if self.record_status_lbl is not None:
                self.record_status_lbl.setText(f"● 录制中 {len(self._record_buffer)} 帧 / {t:.1f}s")



def build_slider_row(parent_layout: QGridLayout, row: int, name: str,
                     lo: float, hi: float, initial: float, ctl: Controller) -> None:
    name_lbl = QLabel(short_label(name))
    name_lbl.setMinimumWidth(120)
    slider = QSlider(Qt.Horizontal)
    slider.setRange(0, SLIDER_RES)
    slider.setValue(ctl.val_to_slider(initial, lo, hi))
    slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    val_lbl = QLabel(f"{initial:+.3f}")
    val_lbl.setMinimumWidth(60)
    val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    val_lbl.setStyleSheet("font-family: Consolas, monospace;")
    range_lbl = QLabel(f"[{lo:+.2f}, {hi:+.2f}]")
    range_lbl.setStyleSheet("color: gray; font-size: 10px;")
    range_lbl.setMinimumWidth(100)
    def on_change(v: int) -> None:
        val_lbl.setText(f"{ctl.slider_to_val(v, lo, hi):+.3f}")
    slider.valueChanged.connect(on_change)
    parent_layout.addWidget(name_lbl, row, 0)
    parent_layout.addWidget(slider, row, 1)
    parent_layout.addWidget(val_lbl, row, 2)
    parent_layout.addWidget(range_lbl, row, 3)
    ctl.sliders[name] = (slider, val_lbl)


def build_ui(window: QMainWindow, ctl: Controller) -> None:
    window.setWindowTitle("灵龙2.0 控制台")
    window.resize(1180, 840)

    central = QWidget()
    window.setCentralWidget(central)
    root_layout = QVBoxLayout(central)
    root_layout.setContentsMargins(8, 8, 8, 8)
    root_layout.setSpacing(8)

    # ---- 工具栏:动作控制 ----
    toolbar1 = QHBoxLayout()
    toolbar1.setSpacing(6)

    toolbar1.addWidget(QLabel("动作:"))
    combo = QComboBox()
    combo.setMinimumWidth(180)
    ctl.motion_combo = combo
    ctl.refresh_motion_combo()
    toolbar1.addWidget(combo)

    btn_play_motion = QPushButton("▶ 播放动作")
    btn_play_motion.setStyleSheet("font-weight: bold;")
    def on_play_motion():
        if ctl.motion_combo is None:
            return
        mid = ctl.motion_combo.currentData()
        if mid:
            ctl.start_motion(mid)
    btn_play_motion.clicked.connect(on_play_motion)
    toolbar1.addWidget(btn_play_motion)

    btn_stop = QPushButton("⏹ 停止")
    btn_stop.clicked.connect(ctl.stop_play)
    toolbar1.addWidget(btn_stop)

    toolbar1.addSpacing(20)

    btn_demo = QPushButton("👋 示例挥手")
    btn_demo.setToolTip("连续循环的挥手示例(老的 PLAY_DEMO)")
    btn_demo.clicked.connect(ctl.start_demo)
    toolbar1.addWidget(btn_demo)

    btn_reset = QPushButton("⟲ 归零")
    btn_reset.clicked.connect(ctl.reset_zero)
    toolbar1.addWidget(btn_reset)

    toolbar1.addSpacing(20)

    toolbar1.addWidget(QLabel("模式:"))
    mode_lbl = QLabel("手动")
    mode_lbl.setStyleSheet("color: #1565c0; font-weight: bold;")
    ctl.mode_label = mode_lbl
    toolbar1.addWidget(mode_lbl)

    toolbar1.addSpacing(10)
    toolbar1.addWidget(QLabel("t:"))
    t_lbl = QLabel("0.00s")
    t_lbl.setStyleSheet("font-family: Consolas, monospace;")
    ctl.t_now_label = t_lbl
    toolbar1.addWidget(t_lbl)

    toolbar1.addStretch()

    btn_save = QPushButton("💾 保存 pose")
    btn_load = QPushButton("📂 加载 pose")
    def on_save():
        path, _ = QFileDialog.getSaveFileName(window, "保存 pose", "pose.json", "JSON (*.json)")
        if path:
            try:
                ctl.save_pose(Path(path))
            except Exception as e:
                ctl.log(f"保存 pose 失败: {e}", "error")
                QMessageBox.warning(window, "保存失败", str(e))
    def on_load():
        path, _ = QFileDialog.getOpenFileName(window, "加载 pose", "", "JSON (*.json)")
        if path:
            try:
                ctl.load_pose(Path(path))
            except Exception as e:
                QMessageBox.warning(window, "加载失败", str(e))
    btn_save.clicked.connect(on_save)
    btn_load.clicked.connect(on_load)
    toolbar1.addWidget(btn_save)
    toolbar1.addWidget(btn_load)

    toolbar1.addSpacing(12)

    # 动作 存/读(整段动作,跨会话持久化)
    btn_save_motion = QPushButton("💾 保存动作")
    btn_save_motion.setToolTip(f"把当前选中的动作存到 {SAVED_MOTIONS_DIR} (跨会话持久化)")
    def on_save_motion():
        if ctl.motion_combo is None:
            return
        mid = ctl.motion_combo.currentData()
        if not mid or mid not in MOTIONS:
            ctl.log("未选中动作,无法保存", "warn")
            return
        try:
            path = SAVED_MOTIONS_DIR / f"{mid}.json"
            ctl.save_motion_to(MOTIONS[mid], path)
            ctl.refresh_motion_combo()
        except Exception as e:
            ctl.log(f"保存动作失败: {e}", "error")
            QMessageBox.warning(window, "保存失败", str(e))
    btn_save_motion.clicked.connect(on_save_motion)
    toolbar1.addWidget(btn_save_motion)

    btn_load_motion = QPushButton("📂 加载动作")
    btn_load_motion.setToolTip("从 JSON 文件读一个动作(覆盖或新加入)")
    def on_load_motion():
        path, _ = QFileDialog.getOpenFileName(window, "加载动作", str(SAVED_MOTIONS_DIR), "JSON (*.json)")
        if not path:
            return
        try:
            mid, motion = ctl.load_motion_from(Path(path))
            MOTIONS[mid] = motion
            ctl.refresh_motion_combo()
            idx = ctl.motion_combo.findData(mid)
            if idx >= 0:
                ctl.motion_combo.setCurrentIndex(idx)
            ctl.set_status(f"已加载动作「{motion.name}」({len(motion.keyframes)} 关键帧)")
        except Exception as e:
            QMessageBox.warning(window, "加载失败", str(e))
    btn_load_motion.clicked.connect(on_load_motion)
    toolbar1.addWidget(btn_load_motion)

    btn_delete_motion = QPushButton("🗑 删除保存")
    btn_delete_motion.setToolTip("删除当前选中动作的已保存文件(下拉框的 ★ 标记消失)")
    def on_delete_motion():
        if ctl.motion_combo is None:
            return
        mid = ctl.motion_combo.currentData()
        if not mid:
            return
        path = SAVED_MOTIONS_DIR / f"{mid}.json"
        if not path.exists():
            ctl.log(f"未找到已保存文件: {path.name}", "warn")
            return
        try:
            path.unlink()
            ctl.log(f"🗑 已删除 {path.name}", "ok")
            ctl.refresh_motion_combo()
        except Exception as e:
            ctl.log(f"删除失败: {e}", "error")
    btn_delete_motion.clicked.connect(on_delete_motion)
    toolbar1.addWidget(btn_delete_motion)

    root_layout.addLayout(toolbar1)

    # ---- toolbar2:时间轴 + 镜像 + 录制 ----
    toolbar2 = QHBoxLayout()
    toolbar2.setSpacing(6)

    toolbar2.addWidget(QLabel("时间轴:"))
    t_scrub = QSlider(Qt.Horizontal)
    t_scrub.setRange(0, 10000)  # 0.00s ~ 100.00s,0.01s 步进
    t_scrub.setValue(0)
    t_scrub.setMinimumWidth(280)
    t_scrub.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    ctl.t_scrub_slider = t_scrub
    def on_t_scrub(v: int) -> None:
        if ctl.motion_combo is None:
            return
        mid = ctl.motion_combo.currentData()
        if not mid or mid not in MOTIONS:
            return
        t = v / 100.0
        motion = MOTIONS[mid]
        # 用户拖时间轴 → 暂停播放
        if ctl._mode == ctl.MODE_MOTION:
            ctl.set_mode(ctl.MODE_MANUAL)
            ctl._motion_id = None
        sample = motion.sample(t)
        for name, val in sample.items():
            if name in ctl.joint_info:
                adr, lo, hi = ctl.joint_info[name]
                clamped = max(lo, min(hi, val))
                ctl.data.qpos[adr] = clamped
                if name in ctl.sliders:
                    slider, lbl = ctl.sliders[name]
                    slider.blockSignals(True)
                    slider.setValue(ctl.val_to_slider(clamped, lo, hi))
                    slider.blockSignals(False)
                    lbl.setText(f"{clamped:+.3f}")
        mujoco.mj_forward(ctl.model, ctl.data)
    t_scrub.valueChanged.connect(on_t_scrub)
    toolbar2.addWidget(t_scrub)

    t_lbl = QLabel("0.00s")
    t_lbl.setMinimumWidth(60)
    t_lbl.setStyleSheet("font-family: Consolas, monospace; color: gray;")
    t_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    toolbar2.addWidget(t_lbl)
    ctl.t_scrub_label = t_lbl

    toolbar2.addSpacing(12)

    # 镜像 pose
    btn_mirror_pose = QPushButton("🪞 镜像 pose")
    btn_mirror_pose.setToolTip("把当前 30 个 slider 的值做 L↔R 镜像(pitch/elbow 同号,roll/yaw 反号)")
    btn_mirror_pose.clicked.connect(ctl.mirror_current_pose)
    toolbar2.addWidget(btn_mirror_pose)

    # 镜像 motion
    btn_mirror_motion = QPushButton("🪞 镜像动作")
    btn_mirror_motion.setToolTip("把下拉框选中的动作做 L↔R 镜像,生成新动作")
    btn_mirror_motion.clicked.connect(ctl.mirror_current_motion)
    toolbar2.addWidget(btn_mirror_motion)

    toolbar2.addSpacing(12)

    # 录制
    btn_record = QPushButton("⏺ 开始录制")
    btn_record.setToolTip("开始/停止录制 — 拖动 sliders 会被记录,停止后保存为新动作")
    def on_record_toggle():
        if ctl._is_recording:
            ctl.stop_recording()
        else:
            ctl.start_recording()
    btn_record.clicked.connect(on_record_toggle)
    ctl.record_btn = btn_record
    toolbar2.addWidget(btn_record)

    rec_status = QLabel("")
    rec_status.setStyleSheet("color: #c62828; font-weight: bold;")
    rec_status.setMinimumWidth(160)
    ctl.record_status_lbl = rec_status
    toolbar2.addWidget(rec_status)

    toolbar2.addStretch()

    root_layout.addLayout(toolbar2)

    # ---- 关节滑块网格(放进一个 QWidget,给 Splitter 用) ----
    slider_container = QWidget()
    slider_layout_outer = QVBoxLayout(slider_container)
    slider_layout_outer.setContentsMargins(0, 0, 0, 0)
    grid = QGridLayout()
    grid.setHorizontalSpacing(8)
    grid.setVerticalSpacing(8)
    for idx, (group_name, joint_names) in enumerate(JOINT_GROUPS):
        col, row = idx % 2, idx // 2
        group = QGroupBox(f"{group_name}  ({len(joint_names)} 个关节)")
        group_layout = QGridLayout(group)
        group_layout.setColumnStretch(1, 1)
        for j, name in enumerate(joint_names):
            adr, lo, hi = ctl.joint_info[name]
            initial = float(ctl.model.qpos0[adr])
            build_slider_row(group_layout, j, name, lo, hi, initial, ctl)
        grid.addWidget(group, row, col)
    grid.setColumnStretch(0, 1)
    grid.setColumnStretch(1, 1)
    slider_layout_outer.addLayout(grid)

    # ---- 日志面板 ----
    log_container = QWidget()
    log_layout = QVBoxLayout(log_container)
    log_layout.setContentsMargins(0, 0, 0, 0)
    log_layout.setSpacing(4)

    log_header = QHBoxLayout()
    log_header.addWidget(QLabel("日志 (可滚动 / 复制):"))
    log_header.addStretch()
    log_count_lbl = QLabel("0 行")
    log_count_lbl.setStyleSheet("color: gray; font-size: 10px;")
    log_header.addWidget(log_count_lbl)
    btn_copy_log = QPushButton("📋 复制全部")
    btn_clear_log = QPushButton("🗑 清空")
    log_header.addWidget(btn_copy_log)
    log_header.addWidget(btn_clear_log)
    log_layout.addLayout(log_header)

    log_view = QPlainTextEdit()
    log_view.setReadOnly(True)
    log_view.setMaximumBlockCount(2000)
    log_view.setStyleSheet(
        "QPlainTextEdit { font-family: Consolas, 'Microsoft YaHei', monospace;"
        " font-size: 11px; background-color: #1e1e1e; color: #d4d4d4;"
        " border: 1px solid #444; }"
    )
    log_view.setMinimumHeight(120)
    ctl.log_view = log_view

    def on_log_count_changed():
        n = log_view.blockCount()
        log_count_lbl.setText(f"{n} 行")
    log_view.blockCountChanged.connect(on_log_count_changed)

    def on_copy_log():
        text = log_view.toPlainText()
        if not text:
            ctl.log("日志为空,无需复制", "warn")
            return
        QApplication.clipboard().setText(text)
        ctl.log(f"已复制 {len(text)} 字符到剪贴板", "ok")

    btn_copy_log.clicked.connect(on_copy_log)
    btn_clear_log.clicked.connect(log_view.clear)
    log_layout.addWidget(log_view)

    # ---- 上下分栏:Splitter 让用户拖拽调高度 ----
    splitter = QSplitter(Qt.Vertical)
    splitter.addWidget(slider_container)
    splitter.addWidget(log_container)
    splitter.setStretchFactor(0, 3)
    splitter.setStretchFactor(1, 1)
    splitter.setSizes([600, 180])
    splitter.setChildrenCollapsible(False)
    root_layout.addWidget(splitter, stretch=1)

    # ---- 状态栏 + 进度条 ----
    status = QStatusBar()
    window.setStatusBar(status)
    status_lbl = QLabel("就绪")
    status_lbl.setStyleSheet("color: #2e7d32; padding: 2px 6px;")
    status.addWidget(status_lbl)
    ctl.status_label = status_lbl

    progress = QProgressBar()
    progress.setRange(0, 100)
    progress.setValue(0)
    progress.setFormat("— / — s")
    progress.setMaximumWidth(280)
    progress.setStyleSheet("QProgressBar { text-align: center; }")
    ctl.motion_progress = progress
    status.addPermanentWidget(progress)

    status.addPermanentWidget(QLabel("提示: 选动作+▶ 播放动作 · 👋示例挥手循环 · 空格=播放选中动作"))

    # ---- 快捷键 ----
    space_sc = QtWidgets.QShortcut(QKeySequence("Space"), window)
    space_sc.activated.connect(on_play_motion)
    r_sc = QtWidgets.QShortcut(QKeySequence("R"), window)
    r_sc.activated.connect(ctl.reset_zero)
    esc_sc = QtWidgets.QShortcut(QKeySequence("Escape"), window)
    esc_sc.activated.connect(ctl.stop_play)


def main() -> None:
    if not MODEL_PATH.exists():
        QMessageBox.critical(None, "模型未找到", f"路径: {MODEL_PATH}")
        return
    try:
        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    except Exception as e:
        QMessageBox.critical(None, "模型加载失败", str(e))
        return

    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)

    app = QApplication.instance() or QApplication(sys.argv)
    ctl = Controller(model, data)
    ctl.log(f"模型加载 OK: {MODEL_PATH.name}  njnt={model.njnt}  nbody={model.nbody}", "ok")

    # 测试模式: --test-motion <id> 自动播放指定动作
    if "--test-motion" in sys.argv:
        try:
            idx = sys.argv.index("--test-motion")
            mid = sys.argv[idx + 1]
            QtCore.QTimer.singleShot(500, lambda: ctl.start_motion(mid))
        except (IndexError, KeyError):
            pass

    window = QMainWindow()
    # 启动时加载已保存动作(覆盖内置同名条目)
    saved = load_all_saved_motions()
    if saved:
        MOTIONS.update(saved)
        ctl.log(f"📦 启动加载了 {len(saved)} 个已保存动作: {', '.join(saved.keys())}", "ok")
    build_ui(window, ctl)
    window.show()

    holder: dict = {}

    def on_window_close(event):
        if holder.get("v") is not None:
            try:
                holder["v"].close()
            except Exception:
                pass
        app.quit()
        event.accept()
    window.closeEvent = on_window_close

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            holder["v"] = viewer
            ctl.set_status(
                f"Viewer 已启动 · njnt={model.njnt} nu={model.nu} · "
                f"动作: {', '.join(m.name for m in MOTIONS.values())}"
            )
            ctl.log("提示: 滑块改 pose · 下拉选动作再 ▶ 播放 · 空格=播放当前动作", "info")

            t0 = time.time()
            def tick() -> None:
                if not viewer.is_running():
                    app.quit()
                    return
                now = time.time()
                t = now - t0  # 面板启动后的秒数(只用于显示)
                if ctl.t_now_label is not None:
                    ctl.t_now_label.setText(f"{t:.2f}s")
                try:
                    if ctl._mode == ctl.MODE_DEMO:
                        ctl.apply_demo(now - ctl._demo_t0)  # 关键:用 now 不用 t
                    elif ctl._mode == ctl.MODE_MOTION and ctl._motion_id:
                        motion = MOTIONS[ctl._motion_id]
                        elapsed = now - ctl._motion_t0  # 关键:用 now 不用 t
                        # 同步时间轴滑块
                        if ctl.t_scrub_slider is not None:
                            ctl.t_scrub_slider.blockSignals(True)
                            ctl.t_scrub_slider.setValue(int(min(elapsed, motion.duration) * 100))
                            ctl.t_scrub_slider.blockSignals(False)
                        if ctl.t_scrub_label is not None:
                            ctl.t_scrub_label.setText(f"{min(elapsed, motion.duration):.2f}s")
                        if ctl.motion_progress is not None:
                            ctl.motion_progress.setValue(min(ctl.motion_progress.maximum(),
                                                              max(0, int(elapsed * 100))))
                            ctl.motion_progress.setFormat(f"{min(max(0, elapsed), motion.duration):.2f} / {motion.duration:.2f} s")
                        still_playing = ctl.apply_motion(elapsed, motion)
                        if not still_playing:
                            ctl._motion_id = None
                    else:
                        ctl.apply_manual()
                    ctl.on_tick_recording()  # 如果在录制,采一帧(只在变化时记)
                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    ctl.set_status(f"播放异常: {e}")
                    ctl.log(f"播放异常: {type(e).__name__}: {e}", "error")
                    ctl.log(tb.rstrip(), "error")
                mujoco.mj_forward(model, data)
                viewer.sync()

            timer = QTimer()
            timer.setInterval(20)
            timer.timeout.connect(tick)
            timer.start()

            app.exec_()
    except Exception as e:
        ctl.log(f"Viewer 启动失败: {type(e).__name__}: {e}", "error")
        QMessageBox.critical(None, "Viewer 启动失败", str(e))


if __name__ == "__main__":
    main()
