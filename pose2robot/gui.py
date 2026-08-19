# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .constants import DEFAULT_MODEL_PATH, DEFAULT_MP_MODEL_PATH, DEFAULT_OUT_DIR
from .pose_mediapipe import LivePoseDetector
from .render_robot import LiveRobotRenderer
from .retarget_ik import UpperBodyIK
from .viz import draw_2d


DEFAULT_VIDEO = Path(r"D:\Downloads\RGB视频数据集.mp4")


def _bgr_to_pixmap(image: np.ndarray) -> QtGui.QPixmap:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimage = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888).copy()
    return QtGui.QPixmap.fromImage(qimage)


def _rgb_to_pixmap(image: np.ndarray) -> QtGui.QPixmap:
    h, w, ch = image.shape
    qimage = QtGui.QImage(image.data, w, h, ch * w, QtGui.QImage.Format_RGB888).copy()
    return QtGui.QPixmap.fromImage(qimage)


class ImagePanel(QGroupBox):
    def __init__(self, title: str):
        super().__init__(title)
        self.pixmap = QtGui.QPixmap()
        self.label = QLabel("No frame")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setMinimumSize(300, 360)
        self.label.setStyleSheet("background:#111;color:#bbb;border:1px solid #333;")
        layout = QVBoxLayout(self)
        layout.addWidget(self.label, 1)

    def set_pixmap(self, pixmap: QtGui.QPixmap) -> None:
        self.pixmap = pixmap
        self._refresh()

    def set_text(self, text: str) -> None:
        self.pixmap = QtGui.QPixmap()
        self.label.setPixmap(QtGui.QPixmap())
        self.label.setText(text)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self) -> None:
        if self.pixmap.isNull():
            return
        scaled = self.pixmap.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label.setPixmap(scaled)


class Pose2RobotLiveViewer(QMainWindow):
    def __init__(self, source: str | None = None):
        super().__init__()
        self.setWindowTitle("pose2robot 实时验证台")
        self.resize(1500, 900)

        self.capture: cv2.VideoCapture | None = None
        self.source = source or str(DEFAULT_VIDEO)
        self.fps = 30.0
        self.frame_index = 0
        self.running = False
        self.last_pose: dict[str, float] | None = None
        self.last_frame_data: dict | None = None
        self.last_error = ""

        self.detector: LivePoseDetector | None = None
        self.ik: UpperBodyIK | None = None
        self.robot: LiveRobotRenderer | None = None
        self._init_engines()

        self.input_panel = ImagePanel("输入: 视频帧 / 摄像头帧")
        self.middle_panel = ImagePanel("中间: 实时人体姿态叠图")
        self.output_panel = ImagePanel("输出: 实时机器人渲染")
        self.info = QPlainTextEdit()
        self.info.setReadOnly(True)
        self.info.setMaximumHeight(230)

        self.source_label = QLabel("")
        self.source_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        open_video = QPushButton("打开视频")
        open_video.clicked.connect(self.open_video)
        open_camera = QPushButton("打开摄像头")
        open_camera.clicked.connect(self.open_camera)
        self.play_button = QPushButton("播放")
        self.play_button.clicked.connect(self.toggle_play)
        step_button = QPushButton("单帧")
        step_button.clicked.connect(self.process_next_frame)
        reset_button = QPushButton("重置")
        reset_button.clicked.connect(self.reset_source)

        top = QHBoxLayout()
        for widget in (open_video, open_camera, self.play_button, step_button, reset_button):
            top.addWidget(widget)
        top.addWidget(self.source_label, 1)

        images = QHBoxLayout()
        images.addWidget(self.input_panel, 1)
        images.addWidget(self.middle_panel, 1)
        images.addWidget(self.output_panel, 1)

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.addLayout(top)
        layout.addLayout(images, 1)
        layout.addWidget(self.info)
        self.setCentralWidget(root)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.process_next_frame)
        self.open_source(self.source)

    def _init_engines(self) -> None:
        try:
            self.detector = LivePoseDetector(DEFAULT_MP_MODEL_PATH)
            self.ik = UpperBodyIK(DEFAULT_MODEL_PATH)
            self.robot = LiveRobotRenderer(DEFAULT_MODEL_PATH)
        except Exception as exc:
            self.last_error = str(exc)

    def open_video(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频",
            str(DEFAULT_OUT_DIR.parent),
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)",
        )
        if selected:
            self.open_source(selected)

    def open_camera(self) -> None:
        self.open_source("0")

    def reset_source(self) -> None:
        self.open_source(self.source)

    def open_source(self, source: str) -> None:
        self.stop()
        if self.capture:
            self.capture.release()
        self.source = source
        capture_source: int | str = int(source) if source.isdigit() else source
        self.capture = cv2.VideoCapture(capture_source)
        if not self.capture.isOpened():
            self.source_label.setText(f"无法打开: {source}")
            self.input_panel.set_text("Cannot open source")
            return
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 30.0)
        if self.fps <= 1.0 or self.fps > 120.0:
            self.fps = 30.0
        self.frame_index = 0
        self.last_pose = None
        self.source_label.setText(f"source: {source}  fps={self.fps:.2f}")
        self.process_next_frame()

    def toggle_play(self) -> None:
        if self.running:
            self.stop()
        else:
            self.running = True
            self.play_button.setText("暂停")
            self.timer.start(max(1, int(1000 / self.fps)))

    def stop(self) -> None:
        self.running = False
        self.timer.stop() if hasattr(self, "timer") else None
        if hasattr(self, "play_button"):
            self.play_button.setText("播放")

    def process_next_frame(self) -> None:
        if not self.capture:
            return
        ok, frame = self.capture.read()
        if not ok:
            self.stop()
            if self.source.isdigit():
                self.info.setPlainText("摄像头读取失败。")
            else:
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.frame_index = 0
                self.info.setPlainText("视频播放结束,已回到开头。")
            return

        started = time.perf_counter()
        self.input_panel.set_pixmap(_bgr_to_pixmap(frame))
        try:
            if not self.detector or not self.ik or not self.robot:
                raise RuntimeError(self.last_error or "实时引擎未初始化")
            frame_data = self.detector.detect_bgr(frame, self.fps)
            overlay = draw_2d(frame, frame_data["pts"])
            pose = self.ik.solve_frame(frame_data, self.last_pose)
            robot_image = self.robot.render(pose)
            self.last_pose = pose
            self.last_frame_data = frame_data
            self.middle_panel.set_pixmap(_bgr_to_pixmap(overlay))
            self.output_panel.set_pixmap(_rgb_to_pixmap(robot_image))
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.info.setPlainText(self._info_text(frame_data, pose, elapsed_ms))
        except Exception as exc:
            self.middle_panel.set_text("Pose failed")
            self.output_panel.set_text("Robot failed")
            self.info.setPlainText(f"frame={self.frame_index}\nerror: {exc}")
        self.frame_index += 1

    def _info_text(self, frame_data: dict, pose: dict[str, float], elapsed_ms: float) -> str:
        visible = [point[3] for point in frame_data["pts"]]
        mean_visibility = float(np.mean(visible)) if visible else 0.0
        lines = [
            f"source: {self.source}",
            f"frame: {frame_data['frame']}  t={frame_data['t']:.2f}s  fps={self.fps:.2f}",
            f"latency: {elapsed_ms:.1f} ms  mean_visibility={mean_visibility:.3f}",
            "",
            "robot joints:",
        ]
        for name, value in pose.items():
            lines.append(f"  {name}: {value:+.3f}")
        return "\n".join(lines)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.stop()
        if self.capture:
            self.capture.release()
        if self.detector:
            self.detector.close()
        if self.robot:
            self.robot.close()
        event.accept()


def main(argv=None) -> int:
    app = QApplication(argv or sys.argv)
    window = Pose2RobotLiveViewer()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
