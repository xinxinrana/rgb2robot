# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from pathlib import Path

import cv2

from .constants import DEFAULT_MP_MODEL_PATH, LANDMARK_NAMES
from .video import video_info

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


class LivePoseDetector:
    def __init__(self, model_path: str | Path = DEFAULT_MP_MODEL_PATH):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError as exc:
            raise RuntimeError("缺少 mediapipe,无法执行人体姿态识别") from exc

        model = Path(model_path)
        if not model.exists():
            raise RuntimeError(f"MediaPipe 模型不存在: {model}")

        self.mp = mp
        self.landmarker = mp_vision.PoseLandmarker.create_from_options(
            mp_vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(model)),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_segmentation_masks=False,
            )
        )
        self.frame_index = 0

    def detect_bgr(self, frame, fps: float = 30.0) -> dict:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(self.frame_index * 1000.0 / max(fps, 1.0))
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks:
            pts = [
                [float(l.x), float(l.y), float(l.z), float(l.visibility)]
                for l in result.pose_landmarks[0]
            ]
        else:
            pts = [[0.0, 0.0, 0.0, 0.0] for _ in LANDMARK_NAMES]

        if result.pose_world_landmarks:
            world = [
                [float(l.x), float(l.y), float(l.z)]
                for l in result.pose_world_landmarks[0]
            ]
        else:
            world = [[0.0, 0.0, 0.0] for _ in LANDMARK_NAMES]

        frame_data = {
            "frame": self.frame_index,
            "t": self.frame_index / max(fps, 1.0),
            "pts": pts,
            "world": world,
        }
        self.frame_index += 1
        return frame_data

    def close(self) -> None:
        self.landmarker.close()


def detect_pose(video_path: str | Path, model_path: str | Path = DEFAULT_MP_MODEL_PATH) -> dict:
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except ImportError as exc:
        raise RuntimeError("缺少 mediapipe,无法执行人体姿态识别") from exc

    model = Path(model_path)
    if not model.exists():
        raise RuntimeError(f"MediaPipe 模型不存在: {model}")

    info = video_info(video_path)
    base = mp_python.BaseOptions(model_asset_path=str(model))
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )

    frames = []
    cap = cv2.VideoCapture(str(video_path))
    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(idx * 1000.0 / info["fps"])
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                pts = [
                    [float(l.x), float(l.y), float(l.z), float(l.visibility)]
                    for l in result.pose_landmarks[0]
                ]
            else:
                pts = [[0.0, 0.0, 0.0, 0.0] for _ in LANDMARK_NAMES]

            if result.pose_world_landmarks:
                world = [
                    [float(l.x), float(l.y), float(l.z)]
                    for l in result.pose_world_landmarks[0]
                ]
            else:
                world = [[0.0, 0.0, 0.0] for _ in LANDMARK_NAMES]

            frames.append({"frame": idx, "t": idx / info["fps"], "pts": pts, "world": world})
            idx += 1
    cap.release()

    return {
        "schema": "pose2robot.pose.v1",
        "video": str(Path(video_path).resolve()),
        "fps": info["fps"],
        "size": [info["width"], info["height"]],
        "frames": len(frames),
        "landmarks": frames,
    }
