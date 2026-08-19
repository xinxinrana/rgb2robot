# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

from .constants import DEFAULT_MODEL_PATH, DEFAULT_MOTION_DIR, DEFAULT_OUT_DIR, DEFAULT_MP_MODEL_PATH
from .io import read_json, write_json
from .pose_mediapipe import detect_pose
from .reports import check_motion, write_joint_csv, write_joint_report
from .render_robot import render_motion_samples, render_pose_set, stitch_comparison
from .retarget import pose_to_joint_rows, rows_to_motion
from .retarget_ik import pose_to_joint_rows_ik
from .robot import load_joint_limits
from .video import extract_frames, video_info
from .viz import visualize_pose


def _session_dir(name: str) -> Path:
    return DEFAULT_OUT_DIR / name


def cmd_video_info(args: argparse.Namespace) -> int:
    info = video_info(args.video)
    print(f"[video] {args.video}")
    print(f"  size={info['width']}x{info['height']} fps={info['fps']:.2f} frames={info['frames']}")
    return 0


def cmd_sample_frames(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir) if args.out_dir else _session_dir(Path(args.video).stem) / "frames"
    written = extract_frames(args.video, out_dir, args.count)
    print(f"[OK] 抽帧 {len(written)} 张 -> {out_dir}")
    return 0


def cmd_detect_pose(args: argparse.Namespace) -> int:
    name = args.name or Path(args.video).stem
    out = Path(args.out) if args.out else _session_dir(name) / "pose.json"
    data = detect_pose(args.video, args.model)
    write_json(out, data)
    print(f"[OK] 姿态识别 -> {out}")
    print(f"     frames={data['frames']} fps={data['fps']:.2f} size={data['size'][0]}x{data['size'][1]}")
    return 0


def cmd_visualize_pose(args: argparse.Namespace) -> int:
    data = read_json(args.pose_json)
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.pose_json).parent / "pose_viz"
    written = visualize_pose(data, out_dir, args.count)
    print(f"[OK] 姿态可视化 {len(written)} 张 -> {out_dir}")
    return 0


def cmd_map_robot(args: argparse.Namespace) -> int:
    pose_data = read_json(args.pose_json)
    name = args.name or Path(args.pose_json).parent.name
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.pose_json).parent / "robot"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, limits = (
        pose_to_joint_rows_ik(pose_data, args.model)
        if args.method == "ik"
        else pose_to_joint_rows(pose_data, args.model)
    )
    csv_path = write_joint_csv(out_dir / "joints.csv", rows)
    report_path = write_joint_report(out_dir / "joints_report.md", pose_data, rows, limits)
    motion = rows_to_motion(rows, name=name, fps=args.fps, smooth_window=args.smooth_window)
    motion_path = write_json(out_dir / "motion.json", motion)
    if args.render:
        rendered = render_motion_samples(motion, args.model, out_dir / "robot_viz", args.render_count)
        print(f"[OK] 机器人渲染 -> {out_dir / 'robot_viz'} ({len(rendered)} 张)")
    if args.export_panel:
        panel_path = DEFAULT_MOTION_DIR / f"{name}.json"
        write_json(panel_path, motion)
        print(f"[OK] panel 动作 -> {panel_path}")

    print(f"[OK] 关节 CSV -> {csv_path}")
    print(f"[OK] 映射报告 -> {report_path}")
    print(f"[OK] Motion -> {motion_path}")
    return 0


def cmd_check_motion(args: argparse.Namespace) -> int:
    motion = read_json(args.motion_json)
    limits = load_joint_limits(args.model)
    errors = check_motion(motion, limits)
    print(f"[motion] {args.motion_json}")
    print(f"  name={motion.get('name', '<missing>')} keyframes={len(motion.get('keyframes', []))}")
    if errors:
        print(f"[FAIL] {len(errors)} 个问题")
        for err in errors[:20]:
            print(f"  {err}")
        return 1
    print("[OK] Motion 合规")
    return 0


def cmd_render_robot(args: argparse.Namespace) -> int:
    motion = read_json(args.motion_json)
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.motion_json).parent / "robot_viz"
    written = render_motion_samples(motion, args.model, out_dir, args.count)
    print(f"[OK] 机器人姿态渲染 {len(written)} 张 -> {out_dir}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    pose_data = read_json(args.pose_json)
    motion = read_json(args.motion_json)
    session = Path(args.out_dir) if args.out_dir else Path(args.pose_json).parent
    human_paths = visualize_pose(pose_data, session / "compare_human", args.count)
    human_2d = [path for path in human_paths if path.name.endswith("_2d.png")]
    robot_paths = render_motion_samples(motion, args.model, session / "compare_robot", args.count)
    combined = stitch_comparison(human_2d, robot_paths, session / "compare")
    print(f"[OK] 人机对比图 {len(combined)} 张 -> {session / 'compare'}")
    return 0


def cmd_joint_atlas(args: argparse.Namespace) -> int:
    limits = load_joint_limits(args.model)
    joints = [
        "waist_yaw_joint", "waist_pitch_joint",
        "head_yaw_joint", "head_pitch_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    ]
    poses = [("zero", {})]
    index_lines = ["# v2 关节视觉 atlas", ""]
    for joint in joints:
        lo, hi = limits[joint]
        neg = max(lo, -abs(args.value))
        pos = min(hi, abs(args.value))
        poses.append((f"{joint}_neg", {joint: neg}))
        poses.append((f"{joint}_pos", {joint: pos}))
        index_lines.append(f"- `{joint}`: neg={neg:+.3f}, pos={pos:+.3f}, limit=[{lo:+.3f}, {hi:+.3f}]")
    out_dir = Path(args.out_dir)
    written = render_pose_set(poses, args.model, out_dir)
    (out_dir / "_index.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    print(f"[OK] 关节 atlas {len(written)} 张 -> {out_dir}")
    print(f"[OK] 索引 -> {out_dir / '_index.md'}")
    return 0


def cmd_gui(args: argparse.Namespace) -> int:
    from .gui import Pose2RobotLiveViewer
    from PyQt5.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv[:1])
    window = Pose2RobotLiveViewer(args.source)
    window.show()
    return app.exec_()


def cmd_run_all(args: argparse.Namespace) -> int:
    name = args.name or Path(args.video).stem
    session = _session_dir(name)
    print(f"[run-all] session={session}")

    info = video_info(args.video)
    print(f"[V0] size={info['width']}x{info['height']} fps={info['fps']:.2f} frames={info['frames']}")

    frames = extract_frames(args.video, session / "frames", args.sample_count)
    print(f"[V0] 抽帧 -> {len(frames)} 张")

    pose_data = detect_pose(args.video, args.model)
    pose_path = write_json(session / "pose.json", pose_data)
    print(f"[A0] 姿态 JSON -> {pose_path}")

    pose_viz = visualize_pose(pose_data, session / "pose_viz", args.sample_count)
    print(f"[A0-check] 姿态可视化 -> {len(pose_viz)} 张")

    rows, limits = (
        pose_to_joint_rows_ik(pose_data, args.robot_model)
        if args.method == "ik"
        else pose_to_joint_rows(pose_data, args.robot_model)
    )
    robot_dir = session / "robot"
    csv_path = write_joint_csv(robot_dir / "joints.csv", rows)
    report_path = write_joint_report(robot_dir / "joints_report.md", pose_data, rows, limits)
    print(f"[A1-check] 关节 CSV -> {csv_path}")
    print(f"[A1-check] 映射报告 -> {report_path}")

    motion = rows_to_motion(rows, name=name, fps=args.fps, smooth_window=args.smooth_window)
    errors = check_motion(motion, limits)
    motion_path = write_json(robot_dir / "motion.json", motion)
    print(f"[A2] Motion -> {motion_path}")
    if errors:
        print(f"[A2-check] FAIL {len(errors)} 个问题")
        for err in errors[:20]:
            print(f"  {err}")
        return 1
    panel_path = DEFAULT_MOTION_DIR / f"{name}.json"
    write_json(panel_path, motion)
    robot_images = render_motion_samples(motion, args.robot_model, robot_dir / "robot_viz", args.sample_count)
    compare_human = visualize_pose(pose_data, session / "compare_human", args.sample_count)
    combined = stitch_comparison(
        [path for path in compare_human if path.name.endswith("_2d.png")],
        robot_images,
        session / "compare",
    )
    print(f"[A2-check] OK")
    print(f"[A2-viz] 机器人图 -> {robot_dir / 'robot_viz'} ({len(robot_images)} 张)")
    print(f"[compare] 人机对比图 -> {session / 'compare'} ({len(combined)} 张)")
    print(f"[export] panel 动作 -> {panel_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="pose2robot v2: 从视频人体姿态到机器人姿态的分步验证流水线")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("video-info", help="读取视频元信息")
    p.add_argument("video")
    p.set_defaults(func=cmd_video_info)

    p = sub.add_parser("sample-frames", help="抽取视频关键帧")
    p.add_argument("video")
    p.add_argument("-n", "--count", type=int, default=8)
    p.add_argument("-o", "--out-dir", default=None)
    p.set_defaults(func=cmd_sample_frames)

    p = sub.add_parser("detect-pose", help="MediaPipe 识别 2D/3D 人体姿态")
    p.add_argument("video")
    p.add_argument("--name", default=None)
    p.add_argument("-o", "--out", default=None)
    p.add_argument("--model", default=str(DEFAULT_MP_MODEL_PATH))
    p.set_defaults(func=cmd_detect_pose)

    p = sub.add_parser("visualize-pose", help="输出 2D 叠图和 3D 三视图")
    p.add_argument("pose_json")
    p.add_argument("-n", "--count", type=int, default=8)
    p.add_argument("-o", "--out-dir", default=None)
    p.set_defaults(func=cmd_visualize_pose)

    p = sub.add_parser("map-robot", help="人体姿态映射到机器人关节和 Motion")
    p.add_argument("pose_json")
    p.add_argument("--name", default=None)
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--smooth-window", type=int, default=7)
    p.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    p.add_argument("--method", choices=["ik", "direct"], default="ik")
    p.add_argument("-o", "--out-dir", default=None)
    p.add_argument("--export-panel", action="store_true")
    p.add_argument("--render", action="store_true", help="同时渲染机器人姿态 PNG")
    p.add_argument("--render-count", type=int, default=12)
    p.set_defaults(func=cmd_map_robot)

    p = sub.add_parser("render-robot", help="把 Motion 抽样渲染成机器人 PNG")
    p.add_argument("motion_json")
    p.add_argument("-n", "--count", type=int, default=12)
    p.add_argument("-o", "--out-dir", default=None)
    p.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    p.set_defaults(func=cmd_render_robot)

    p = sub.add_parser("compare", help="生成原视频人体 2D 骨架与机器人姿态对比图")
    p.add_argument("pose_json")
    p.add_argument("motion_json")
    p.add_argument("-n", "--count", type=int, default=12)
    p.add_argument("-o", "--out-dir", default=None)
    p.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("joint-atlas", help="渲染机器人上半身关节正负方向 atlas")
    p.add_argument("-o", "--out-dir", default=str(DEFAULT_OUT_DIR / "joint_atlas"))
    p.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    p.add_argument("--value", type=float, default=0.8)
    p.set_defaults(func=cmd_joint_atlas)

    p = sub.add_parser("gui", help="启动三栏图像验证 GUI")
    p.add_argument("--source", default=r"D:\Downloads\RGB视频数据集.mp4",
                   help="视频路径或摄像头编号,例如 0")
    p.set_defaults(func=cmd_gui)

    p = sub.add_parser("check-motion", help="校验 Motion JSON")
    p.add_argument("motion_json")
    p.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    p.set_defaults(func=cmd_check_motion)

    p = sub.add_parser("run-all", help="端到端执行并保留全部中间产物")
    p.add_argument("video")
    p.add_argument("--name", default=None)
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--smooth-window", type=int, default=7)
    p.add_argument("--sample-count", type=int, default=8)
    p.add_argument("--model", default=str(DEFAULT_MP_MODEL_PATH))
    p.add_argument("--robot-model", default=str(DEFAULT_MODEL_PATH))
    p.add_argument("--method", choices=["ik", "direct"], default="ik")
    p.set_defaults(func=cmd_run_all)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
