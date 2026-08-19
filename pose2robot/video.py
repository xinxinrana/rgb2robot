# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def video_info(video_path: str | Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return {"fps": fps, "width": width, "height": height, "frames": frames}


def sample_indices(total_frames: int, count: int) -> list[int]:
    if total_frames <= 0:
        return []
    if count <= 1:
        return [0]
    return sorted(set(int(round(v)) for v in np.linspace(0, total_frames - 1, count)))


def extract_frames(video_path: str | Path, out_dir: str | Path, count: int = 8) -> list[Path]:
    video = Path(video_path)
    info = video_info(video)
    indices = sample_indices(info["frames"], count)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    written: list[Path] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        path = out / f"{video.stem}_f{idx:05d}_t{idx / info['fps']:.2f}.png"
        ok, encoded = cv2.imencode(".png", frame)
        if not ok:
            continue
        encoded.tofile(str(path))
        written.append(path)
    cap.release()
    return written


def read_frame(video_path: str | Path, frame_index: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None
