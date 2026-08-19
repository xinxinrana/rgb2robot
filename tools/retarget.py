# -*- coding: utf-8 -*-
"""A1+A2: MediaPipe pose.json → 机器人关节角 → 平滑 → Motion JSON。

输入:out/<name>.pose.json(MediaPipe PoseLandmarker 输出格式)
输出:sim/saved_motions/<name>.json(panel.py 可直接 ▶ 播放)

只做上半身(头 + 腰 + 双臂 pitch/roll/elbow)。其它关节(下半身、wrist、yaw)
先锁 0,MVP 阶段够用。
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

# MediaPipe PoseLandmarker 33 关键点
LM = {
    "nose": 0,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow":    13, "right_elbow":    14,
    "left_wrist":    15, "right_wrist":    16,
    "left_hip":      23, "right_hip":      24,
}

def _v(p, q):
    return np.array([q[0] - p[0], q[1] - p[1], q[2] - p[2]], dtype=float)


def _norm(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v * 0


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


def _retarget_one(pts: dict) -> dict:
    sh_L = np.array(pts["left_shoulder"])
    sh_R = np.array(pts["right_shoulder"])
    el_L = np.array(pts["left_elbow"])
    el_R = np.array(pts["right_elbow"])
    wr_L = np.array(pts["left_wrist"])
    wr_R = np.array(pts["right_wrist"])
    hip_L = np.array(pts["left_hip"])
    hip_R = np.array(pts["right_hip"])
    nose = np.array(pts["nose"])

    shoulder_mid = (sh_L + sh_R) / 2
    hip_mid = (hip_L + hip_R) / 2
    torso = shoulder_mid - hip_mid

    # 头:头向量 = (nose - 肩中点)
    head_vec = nose - shoulder_mid
    # head_yaw:头的水平偏航。约定 +head_yaw = 头向机器人左手(画面右)= -X
    head_yaw = -math.atan2(-head_vec[0], -head_vec[2])
    # head_pitch: +pitch = 低头。头向量 Y > 0 抬头 → pitch 负
    head_pitch = -math.atan2(-head_vec[1],
                             math.sqrt(head_vec[0]**2 + head_vec[2]**2))

    # 腰:躯干方向
    # waist_pitch: +弯腰(前倾)。躯干指向上(肩在髋上),前倾 = 肩在髋前 = +Z
    waist_pitch = -math.atan2(torso[2], abs(torso[1]) + 0.01)
    # waist_yaw: +向机器人左手扭 = 肩相对髋向 -X 偏
    waist_yaw = -math.atan2(-(shoulder_mid[0] - hip_mid[0]),
                            abs(torso[1]) + 0.01)

    def arm_angles(side: str):
        sh, el, wr, hip = (
            (sh_L, el_L, wr_L, hip_L) if side == "L"
            else (sh_R, el_R, wr_R, hip_R)
        )
        u = _norm(_v(sh, el))
        up = np.array([0.0, 1.0, 0.0])
        forward = np.array([0.0, 0.0, -1.0])  # 相机方向 = 人/机器人前方
        # 机器人右手=+X,左手=-X。MediaPipe 命名:人正对相机,MP right = 人的左手 = 画面右 = 机器人左手
        # 等等:画面右 ≠ 机器人左手。看 panel.py 文档:
        #   画面里"右" = 机器人的左手侧(镜像)
        #   画面里"左" = 机器人的右手侧
        # MediaPipe:人正对相机,MP "left" = 人的右手(因为镜像)。
        # 那画面左 = MP left?  ←  画面左 = 机器人右手 = 人的左手 = MP right
        # 所以:MP left → 画面右 → 机器人左
        #     MP right → 画面左 → 机器人右
        # 因此:side='L'(机器人左)= MP right
        #     side='R'(机器人右)= MP left
        right = np.array([-1.0, 0.0, 0.0] if side == "L" else [1.0, 0.0, 0.0])

        uf = float(np.dot(u, forward))
        uu = float(np.dot(u, up))
        ur = float(np.dot(u, right))

        # 约定:shoulder_pitch > 0 向后,-向前
        # 垂臂:uf=0, uu=-1 → pitch=0 ✓
        # 前推:uf=+1, uu=0 → pitch 应为负 → atan2(uf, -uu) = atan2(1,0)=π/2
        # 所以取反:pitch = -atan2(uf, -uu)
        pitch = -math.atan2(uf, -uu)

        # 约定:shoulder_roll > 0 抬高
        roll = math.atan2(ur, -uu)

        # elbow:上臂与小臂的夹角
        f_ = _norm(_v(el, wr))
        cos_ang = float(np.clip(np.dot(u, f_), -1.0, 1.0))
        ang = math.acos(cos_ang)
        elbow = (math.pi / 2 - ang) / (math.pi / 2)
        elbow = max(-1.0, min(1.0, elbow))
        return pitch, roll, elbow

    pitch_L, roll_L, elbow_L = arm_angles("L")
    pitch_R, roll_R, elbow_R = arm_angles("R")

    out = {
        "head_yaw_joint":     _clip(head_yaw,   -1.57,  1.57),
        "head_pitch_joint":   _clip(head_pitch, -0.79,  0.79),
        "waist_yaw_joint":    _clip(waist_yaw,  -2.62,  2.62),
        "waist_pitch_joint":  _clip(waist_pitch,-0.52,  0.52),
        "left_shoulder_pitch_joint":  _clip(pitch_L, -2.97, 2.97),
        "left_shoulder_roll_joint":   _clip(roll_L,  -0.09, 3.05),
        "left_elbow_joint":           _clip(elbow_L, -0.79, 1.57),
        "right_shoulder_pitch_joint": _clip(pitch_R, -2.97, 2.97),
        "right_shoulder_roll_joint":  _clip(roll_R,  -3.05, 0.09),
        "right_elbow_joint":          _clip(elbow_R, -0.79, 1.57),
    }
    return out


def _smooth(arr: np.ndarray, window: int = 11) -> np.ndarray:
    from scipy.signal import savgol_filter
    if window < 3 or len(arr) < window:
        return arr
    w = window if window % 2 == 1 else window + 1
    return savgol_filter(arr, window_length=w, polyorder=3, axis=0)


def _downsample(keyframes: List[dict], src_fps: float, target_fps: float) -> List[dict]:
    if target_fps <= 0 or target_fps >= src_fps:
        return keyframes
    step = src_fps / target_fps
    out = []
    i = 0.0
    while int(round(i)) < len(keyframes):
        out.append(keyframes[int(round(i))])
        i += step
    return out


def retarget(pose_data: dict, target_fps: float = 10.0) -> dict:
    src_fps = float(pose_data["fps"])
    frames = pose_data["landmarks"]
    if not frames:
        raise SystemExit("pose.json 里没有 frames")
    joint_names = [
        "head_yaw_joint", "head_pitch_joint",
        "waist_yaw_joint", "waist_pitch_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_elbow_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_elbow_joint",
    ]
    per_joint = {n: [] for n in joint_names}
    times: List[float] = []
    for f in frames:
        world = f["world"]
        pts = {
            "nose":            world[0],
            "left_shoulder":   world[LM["left_shoulder"]],
            "right_shoulder":  world[LM["right_shoulder"]],
            "left_elbow":      world[LM["left_elbow"]],
            "right_elbow":     world[LM["right_elbow"]],
            "left_wrist":      world[LM["left_wrist"]],
            "right_wrist":     world[LM["right_wrist"]],
            "left_hip":        world[LM["left_hip"]],
            "right_hip":       world[LM["right_hip"]],
        }
        ang = _retarget_one(pts)
        for n in joint_names:
            per_joint[n].append(ang[n])
        times.append(f["t"])

    win = min(11, max(3, (len(times) // 5) | 1))
    smoothed = {}
    for n, vals in per_joint.items():
        smoothed[n] = _smooth(np.asarray(vals, dtype=float), window=win).tolist()

    keyframes = []
    for i, t in enumerate(times):
        pose = {n: float(smoothed[n][i]) for n in joint_names}
        keyframes.append({"t": float(t), "pose": pose})

    if target_fps > 0 and target_fps < src_fps:
        keyframes = _downsample(keyframes, src_fps, target_fps)

    return {
        "id": "<from-video>",
        "name": "video_motion",
        "keyframes": keyframes,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="A1+A2: pose.json → 机器人 Motion JSON")
    ap.add_argument("pose_json", help="MediaPipe 风格 pose.json 路径")
    ap.add_argument("-o", "--out", default=None,
                    help="Motion JSON 输出路径(默认 sim/saved_motions/<stem>.json)")
    ap.add_argument("--fps", type=float, default=10.0,
                    help="降采样目标 fps(0=不降)")
    ap.add_argument("--name", default=None, help="动作名(默认 <stem>)")
    args = ap.parse_args(argv)

    in_path = Path(args.pose_json)
    if not in_path.exists():
        print(f"[ERR] 不存在: {in_path}", file=sys.stderr)
        return 2
    print(f"[A1+A2] {in_path}")
    pose_data = json.loads(in_path.read_text(encoding="utf-8"))
    motion = retarget(pose_data, target_fps=args.fps)
    if args.name:
        motion["name"] = args.name

    out_path = (Path(args.out) if args.out
                else Path("sim/saved_motions") / (in_path.name.replace(".pose.json", "") + ".json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(motion, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] {out_path}  {len(motion['keyframes'])} 关键帧  "
          f"({out_path.stat().st_size/1024:.1f} KB)")
    for tag, kf in [("first", motion["keyframes"][0]),
                    ("last",  motion["keyframes"][-1])]:
        sample = ", ".join(f"{n}={v:+.2f}" for n, v in list(kf["pose"].items())[:4])
        print(f"  t={kf['t']:.2f}s  {sample} ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
