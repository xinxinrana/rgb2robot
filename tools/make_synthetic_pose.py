# -*- coding: utf-8 -*-
"""A0 测试数据生成:合成 MediaPipe 格式的 pose.json(模拟"挥手"动作)。

不依赖真人素材 / MediaPipe。直接生成与 MediaPipe PoseLandmarker 同结构
(33 关键点,x/y/z/visibility + world x/y/z)的 JSON,用于验证 A1/A2 pipeline。
输出 out/_test_wave.pose.json
"""
from __future__ import annotations
import json, math, os
from pathlib import Path

OUT = "out/_test_wave.pose.json"
FPS = 30.0
DUR = 4.0
N = int(FPS * DUR)

NAMES = [
    "nose","left_eye_inner","left_eye","left_eye_outer",
    "right_eye_inner","right_eye","right_eye_outer",
    "left_ear","right_ear","mouth_left","mouth_right",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_pinky","right_pinky",
    "left_index","right_index","left_thumb","right_thumb",
    "left_hip","right_hip","left_knee","right_knee",
    "left_ankle","right_ankle","left_heel","right_heel",
    "left_foot_index","right_foot_index",
]
assert len(NAMES) == 33

# T-pose 基线(米),X 向右,Y 向上,Z 向相机外
BASE = {
    "nose":              ( 0.00,  0.00,  0.00),
    "left_eye_inner":    (-0.03,  0.03,  0.02),
    "left_eye":          (-0.04,  0.03,  0.02),
    "left_eye_outer":    (-0.06,  0.03,  0.02),
    "right_eye_inner":   ( 0.03,  0.03,  0.02),
    "right_eye":         ( 0.04,  0.03,  0.02),
    "right_eye_outer":   ( 0.06,  0.03,  0.02),
    "left_ear":          (-0.10,  0.00,  0.00),
    "right_ear":         ( 0.10,  0.00,  0.00),
    "mouth_left":        (-0.03, -0.05,  0.02),
    "mouth_right":       ( 0.03, -0.05,  0.02),
    "left_shoulder":     (-0.20, -0.05,  0.00),
    "right_shoulder":    ( 0.20, -0.05,  0.00),
    "left_elbow":        (-0.40, -0.05,  0.00),
    "right_elbow":       ( 0.40, -0.05,  0.00),
    "left_wrist":        (-0.60, -0.05,  0.00),
    "right_wrist":       ( 0.60, -0.05,  0.00),
    "left_pinky":        (-0.62, -0.05,  0.00),
    "right_pinky":       ( 0.62, -0.05,  0.00),
    "left_index":        (-0.61, -0.05,  0.00),
    "right_index":       ( 0.61, -0.05,  0.00),
    "left_thumb":        (-0.59, -0.05,  0.00),
    "right_thumb":       ( 0.59, -0.05,  0.00),
    "left_hip":          (-0.15, -0.45,  0.00),
    "right_hip":         ( 0.15, -0.45,  0.00),
    "left_knee":         (-0.15, -0.80,  0.00),
    "right_knee":        ( 0.15, -0.80,  0.00),
    "left_ankle":        (-0.15, -1.10,  0.00),
    "right_ankle":       ( 0.15, -1.10,  0.00),
    "left_heel":         (-0.17, -1.12,  0.00),
    "right_heel":        ( 0.17, -1.12,  0.00),
    "left_foot_index":   (-0.13, -1.12,  0.00),
    "right_foot_index":  ( 0.13, -1.12,  0.00),
}


def lift_arm(pts: dict, side: str, t: float) -> dict:
    """绕肩把整条臂从身侧举到头顶(在 XY 面内绕 Z 摆动)。
    side='right' 用 right_*,side='left' 用 left_*。
    """
    raise_amt = 0.5 + 0.5 * math.sin(2 * math.pi * t * 0.75)
    theta = -math.pi / 2 + raise_amt * math.pi  # 垂下→头顶
    sh = pts[f"{side}_shoulder"]
    out = dict(pts)
    for k in (f"{side}_elbow", f"{side}_wrist",
              f"{side}_pinky", f"{side}_index", f"{side}_thumb"):
        x, y, z = pts[k]
        rx, ry, rz = x - sh[0], y - sh[1], z - sh[2]
        nx = math.cos(theta) * rx - math.sin(theta) * ry
        ny = math.sin(theta) * rx + math.cos(theta) * ry
        out[k] = (sh[0] + nx, sh[1] + ny, sh[2] + rz)
    return out


def frame_at(i: int) -> dict:
    t = i / FPS
    pts = dict(BASE)
    sway = 0.02 * math.sin(2 * math.pi * t * 0.5)
    for k in list(pts.keys()):
        x, y, z = pts[k]
        pts[k] = (x, y + sway, z)
    # 挥右手(画面右 = MediaPipe right = 机器人 left,因为镜像)
    # 真实人正对相机时:画面右 = 人的左手(MediaPipe 把它叫 left)
    # 我们假设视频里人正对相机 → MediaPipe "right" 对应机器人 left
    # 但用户视角下"挥手"用哪只手都行,这里让两侧都稍动,主要动 MediaPipe right(=机器人 left)
    pts = lift_arm(pts, "right", t)
    head_nod = 0.05 * math.sin(2 * math.pi * t * 0.75)
    for k in ("nose","left_eye_inner","left_eye","left_eye_outer",
              "right_eye_inner","right_eye","right_eye_outer",
              "left_ear","right_ear","mouth_left","mouth_right"):
        x, y, z = pts[k]
        pts[k] = (x, y + head_nod * 0.5, z + head_nod * 0.2)

    out_pts = []; out_world = []
    for n in NAMES:
        x, y, z = pts[n]
        out_pts.append([x, -y, -z, 0.95])
        out_world.append([x, y, z])
    return {"t": t, "pts": out_pts, "world": out_world}


def main():
    os.makedirs("out", exist_ok=True)
    frames = [frame_at(i) for i in range(N)]
    data = {
        "video": "<synthetic: tools/make_synthetic_pose.py>",
        "fps": FPS, "frames": N, "size": [640, 480],
        "landmarks": frames,
    }
    Path(OUT).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] {OUT}  {N} 帧 @ {FPS} fps  ({Path(OUT).stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
