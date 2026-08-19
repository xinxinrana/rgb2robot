# -*- coding: utf-8 -*-
"""A0: mp4 → MediaPipe PoseLandmarker 33 关键点 → JSON.

输出 out/<video>.pose.json,结构:
  {
    "video": ".../xxx.mp4",
    "fps":   <float>,
    "frames": <int>,
    "size":  [W, H],
    "landmarks": [
      {  # 一帧
        "t": <float 秒>,
        "pts": [ [x, y, z, visibility], ... 33 项 ],  # 世界/图像归一化坐标
        "world": [ [x, y, z], ... 33 项 ]             # 3D 世界坐标(米)
      },
      ...
    ]
  }
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
from typing import List

import cv2
import numpy as np

# 关掉 TF 噪音
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# MediaPipe PoseLandmarker 33 点索引(与旧 solutions API 一致)
POSE_LMS = [
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
N_LM = len(POSE_LMS)
assert N_LM == 33


def _make_landmarker(model_path: str) -> "mp_vision.PoseLandmarker":
    base = mp_python.BaseOptions(model_asset_path=model_path)
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=base,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    return mp_vision.PoseLandmarker.create_from_options(opts)


def _extract(video_path: str, model_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  视频: {video_path}\n  fps={fps:.2f}  size={W}x{H}  frames={total}")

    lm = _make_landmarker(model_path)
    out_frames: List[dict] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # MediaPipe 用 mp.Image,先 BGR→RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # 时间戳必须毫秒,单调递增
        ts_ms = int(idx * 1000.0 / fps)
        result = lm.detect_for_video(mp_img, ts_ms)
        if result.pose_landmarks:
            lms = result.pose_landmarks[0]  # 取第一个人
            pts = [[float(l.x), float(l.y), float(l.z), float(l.visibility)] for l in lms]
        else:
            pts = [[0.0, 0.0, 0.0, 0.0]] * N_LM
        if result.pose_world_landmarks:
            wlms = result.pose_world_landmarks[0]
            world = [[float(l.x), float(l.y), float(l.z)] for l in wlms]
        else:
            world = [[0.0, 0.0, 0.0]] * N_LM
        out_frames.append({"t": idx / fps, "pts": pts, "world": world})
        idx += 1
        if idx % 30 == 0:
            print(f"  ... {idx}/{total}", flush=True)
    cap.release()
    lm.close()
    return {
        "video": str(Path(video_path).resolve()),
        "fps": fps,
        "frames": idx,
        "size": [W, H],
        "landmarks": out_frames,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="A0: 视频 → MediaPipe 33 关键点 JSON")
    ap.add_argument("video", help="输入 mp4 文件")
    ap.add_argument("-m", "--model", default="tools/models/pose_landmarker.task",
                    help="MediaPipe PoseLandmarker .task 模型路径")
    ap.add_argument("-o", "--out-dir", default="out", help="JSON 输出目录")
    args = ap.parse_args(argv)

    if not os.path.exists(args.model):
        print(f"[ERR] 模型文件不存在: {args.model}", file=sys.stderr)
        print("      请先跑:python -c \"import urllib.request; urllib.request.urlretrieve("
              "'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task',"
              " 'tools/models/pose_landmarker.task')\"", file=sys.stderr)
        return 2
    if not os.path.exists(args.video):
        print(f"[ERR] 视频文件不存在: {args.video}", file=sys.stderr)
        return 2

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[A0] {args.video} → {args.out_dir}")
    data = _extract(args.video, args.model)
    out_path = Path(args.out_dir) / (Path(args.video).stem + ".pose.json")
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] {out_path}  ({len(data['landmarks'])} 帧, "
          f"{out_path.stat().st_size/1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
