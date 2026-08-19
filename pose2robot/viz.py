# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .constants import LM, POSE_EDGES
from .video import read_frame, sample_indices


def draw_2d(frame: np.ndarray, pts: list[list[float]]) -> np.ndarray:
    h, w = frame.shape[:2]
    out = frame.copy()
    pixels = [(int(p[0] * w), int(p[1] * h), float(p[3])) for p in pts]
    for a_name, b_name in POSE_EDGES:
        a = LM[a_name]
        b = LM[b_name]
        if pixels[a][2] < 0.2 or pixels[b][2] < 0.2:
            continue
        cv2.line(out, pixels[a][:2], pixels[b][:2], (20, 160, 255), 2, cv2.LINE_AA)
    for x, y, visibility in pixels:
        color = (20, 180, 40) if visibility >= 0.5 else (0, 0, 220)
        cv2.circle(out, (x, y), 3, color, -1, cv2.LINE_AA)
    return out


def _project(points: np.ndarray, axes: tuple[int, int], size: int = 420) -> np.ndarray:
    canvas = np.full((size, size, 3), 248, dtype=np.uint8)
    xy = points[:, axes].astype(float)
    center = np.nanmean(xy, axis=0)
    xy = xy - center
    scale = np.nanmax(np.abs(xy))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    return (xy / scale * (size * 0.38) + size / 2).astype(int), canvas


def draw_3d(world: list[list[float]]) -> np.ndarray:
    points = np.asarray(world, dtype=float)
    views = []
    for title, axes in [("front X/Y", (0, 1)), ("side Z/Y", (2, 1)), ("top X/Z", (0, 2))]:
        pixels, canvas = _project(points, axes)
        for a_name, b_name in POSE_EDGES:
            cv2.line(canvas, tuple(pixels[LM[a_name]]), tuple(pixels[LM[b_name]]), (80, 120, 220), 2, cv2.LINE_AA)
        for pixel in pixels:
            cv2.circle(canvas, tuple(pixel), 3, (40, 40, 40), -1, cv2.LINE_AA)
        cv2.putText(canvas, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 1)
        views.append(canvas)
    return np.concatenate(views, axis=1)


def visualize_pose(pose_data: dict, out_dir: str | Path, count: int = 8) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames = pose_data["landmarks"]
    indices = sample_indices(len(frames), count)
    video_path = Path(pose_data.get("video", ""))
    width, height = pose_data.get("size", [640, 480])
    written: list[Path] = []

    for idx in indices:
        item = frames[idx]
        frame = read_frame(video_path, item.get("frame", idx)) if video_path.exists() else None
        if frame is None:
            frame = np.full((height, width, 3), 245, dtype=np.uint8)
        stem = f"f{idx:05d}_t{item['t']:.2f}"
        path_2d = out / f"{stem}_2d.png"
        path_3d = out / f"{stem}_3d.png"
        for path, image in [(path_2d, draw_2d(frame, item["pts"])), (path_3d, draw_3d(item["world"]))]:
            ok, encoded = cv2.imencode(".png", image)
            if not ok:
                continue
            encoded.tofile(str(path))
        written.extend([path_2d, path_3d])
    return written
