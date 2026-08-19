# -*- coding: utf-8 -*-
from __future__ import annotations

"""关节语义校准脚本。

渲染一组测试姿态,生成 PNG 到 calibration/ 目录(每张图同时存到 calibration/<label>.png
和 calibration/_index.txt 一份索引)。用户打开 PNG,按图名里的问题逐张描述实际看到的样子,
发回给我,我据此建立 "关节值 -> 视觉效果" 的映射表,再回头重做动作设计。

用法:
    .venv\\Scripts\\python.exe sim\\calibrate.py
    .venv\\Scripts\\python.exe sim\\calibrate.py --label A0  # 只渲染某一张(调试用)
"""
import sys

# 强制 stdout 用 UTF-8,避免 Windows GBK 控制台编码报错
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import mujoco
from mujoco import Renderer

ROOT = Path(__file__).resolve().parent.parent
MODEL = Path(r"D:\Code\Other\0809rebot\model\LingLong2.0\LingLong2.0.urdf")
OUT_DIR = ROOT / "calibration"

# 测试姿态列表:(label, 我提的问题, {joint_name: value, ...})
# value 写 None 表示"用 model.qpos0 当基线再改这一个关节"
# 也接受 "base_z"/"base_x"/"base_y" 写到 data.qpos[0..2]
TEST_POSES: list[Tuple[str, str, Optional[Dict[str, float]]]] = [
    # ===== A. 基线 =====
    (
        "A0_default",
        "A0 [基线]: 用 model.qpos0 当家底,其它关节不动 —— 机器人在画面里是什么姿势?\n"
        "     站着 / 跪着 / 躺平 / T-pose / 双臂下垂 / 双臂举起 ... 越细越好。",
        None,
    ),

    # ===== B. 单关节轴向(头) =====
    (
        "B1_head_yaw_pos",
        "B1 [头偏航 +1.0]: 头朝哪边转?我眼里机器人的左/右,还是你眼里(面对机器人)的左/右?",
        {"head_yaw_joint": 1.0},
    ),
    (
        "B2_head_yaw_neg",
        "B2 [头偏航 -1.0]: 头朝哪边转?(和 B1 比一下)",
        {"head_yaw_joint": -1.0},
    ),
    (
        "B3_head_pitch_pos",
        "B3 [头俯仰 +0.5]: 头抬起(仰)还是低下(俯)?",
        {"head_pitch_joint": 0.5},
    ),
    (
        "B4_head_pitch_neg",
        "B4 [头俯仰 -0.5]: 头抬起(仰)还是低下(俯)?",
        {"head_pitch_joint": -0.5},
    ),

    # ===== B. 单关节轴向(腰) =====
    (
        "B5_waist_yaw_pos",
        "B5 [腰偏航 +1.0]: 整个身体(从腰开始)朝哪边转?",
        {"waist_yaw_joint": 1.0},
    ),
    (
        "B6_waist_yaw_neg",
        "B6 [腰偏航 -1.0]: 整个身体朝哪边转?(和 B5 比)",
        {"waist_yaw_joint": -1.0},
    ),
    (
        "B7_waist_pitch_pos",
        "B7 [腰俯仰 +0.4]: 身体前弯(鞠躬)还是后仰(挺胸)?",
        {"waist_pitch_joint": 0.4},
    ),
    (
        "B8_waist_pitch_neg",
        "B8 [腰俯仰 -0.4]: 身体前弯还是后仰?(和 B7 比)",
        {"waist_pitch_joint": -0.4},
    ),

    # ===== B. 单关节轴向(左肩) =====
    (
        "B9_L_shoulder_pitch_pos",
        "B9 [左肩俯仰 +1.0]: 左臂往哪个方向?前/后/上/侧?",
        {"left_shoulder_pitch_joint": 1.0},
    ),
    (
        "B10_L_shoulder_pitch_neg",
        "B10 [左肩俯仰 -1.0]: 左臂往哪个方向?(和 B9 比)",
        {"left_shoulder_pitch_joint": -1.0},
    ),
    (
        "B11_L_shoulder_roll_pos",
        "B11 [左肩翻滚 +2.0]: 左臂往哪个方向?是不是从身侧抬到高处?",
        {"left_shoulder_roll_joint": 2.0},
    ),
    (
        "B12_L_shoulder_roll_neg",
        "B12 [左肩翻滚 -0.05(接近下限)]: 左臂往哪个方向?是不是贴着身侧 / 略前伸?",
        {"left_shoulder_roll_joint": -0.05},
    ),

    # ===== B. 单关节轴向(肘、膝) =====
    (
        "B13_L_elbow_pos",
        "B13 [左肘 +1.0]: 左手肘怎么变?弯曲 / 反向 / 不动?",
        {"left_elbow_joint": 1.0},
    ),
    (
        "B14_L_knee_pos",
        "B14 [左膝 +1.0]: 左腿怎么变?弯曲(下蹲)还是反向(后抬)?",
        {"left_knee_joint": 1.0},
    ),

    # ===== C. T-pose 验证 =====
    (
        "C1_T_pose",
        "C1 [T-pose 试猜]: 左肩 roll=+1.57, 右肩 roll=-1.57, 其它不动 —— 双臂是不是展开成 'T' 字?\n"
        "     如果不是,实际看到的是啥?哪个臂抬高了?哪个还垂着?",
        {
            "left_shoulder_roll_joint": 1.57,
            "right_shoulder_roll_joint": -1.57,
        },
    ),

    # ===== D. 关键动作猜测(我现有动作里用的) =====
    (
        "D1_arms_up",
        "D1 [我的 '举高' 猜测]: 双肩 pitch=-1.57, 双肩 roll=±0.78 —— 双手是不是举到头上了?\n"
        "     如果不是,实际看到的是啥?手在身体哪个方位?",
        {
            "left_shoulder_pitch_joint": -1.57,
            "right_shoulder_pitch_joint": -1.57,
            "left_shoulder_roll_joint": 0.78,
            "right_shoulder_roll_joint": -0.78,
        },
    ),
    (
        "D2_bow",
        "D2 [我的 '鞠躬' 猜测]: 腰 pitch=+0.5, 头 pitch=+0.3 —— 是不是弯腰低头?",
        {
            "waist_pitch_joint": 0.5,
            "head_pitch_joint": 0.3,
        },
    ),
    (
        "D3_sway_left",
        "D3 [我的 '摆胯左' 猜测]: 腰 yaw=-0.3 —— 是不是上身 / 头转向机器人左肩那边?",
        {
            "waist_yaw_joint": -0.3,
        },
    ),
    (
        "D4_hip_yaw_demo",
        "D4 [顺带验证]: 把 qpos[2](其实是 left_hip_yaw)=0.3 —— 左大腿是不是横着转?这相当于 panel.py 以前 'base_z=0.3' 实际产生的效果。",
        {
            "left_hip_yaw_joint": 0.3,
        },
    ),
    (
        "D5_push_palm",
        "D5 [我的 '推掌' 猜测]: 右肩 pitch=-1.0, 右肩 roll=-0.5, 肘=0 —— 右臂是不是前伸?掌心是不是朝前?",
        {
            "right_shoulder_pitch_joint": -1.0,
            "right_shoulder_roll_joint": -0.5,
            "right_elbow_joint": 0.0,
        },
    ),
]


def apply_pose(model: mujoco.MjModel, data: mujoco.MjData, joint_dict: Optional[Dict[str, float]]) -> None:
    """把 joint_dict 写到 data.qpos,其它位置保留 model.qpos0 的值。"""
    data.qpos[:] = model.qpos0
    if not joint_dict:
        return
    for name, val in joint_dict.items():
        if name == "base_z":
            data.qpos[2] = float(val)
            continue
        if name == "base_x":
            data.qpos[0] = float(val)
            continue
        if name == "base_y":
            data.qpos[1] = float(val)
            continue
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            print(f"  [警告] 关节 '{name}' 不存在,跳过", file=sys.stderr)
            continue
        adr = int(model.joint(jid).qposadr[0])
        lo, hi = float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1])
        clamped = max(lo, min(hi, float(val)))
        if clamped != val:
            print(f"  [限位] {name}: {val:.3f} -> 截到 {clamped:.3f}(范围 [{lo:.3f}, {hi:.3f}])", file=sys.stderr)
        data.qpos[adr] = clamped


def make_renderer(model: mujoco.MjModel) -> Renderer:
    # 默认 framebuffer 上限 480x480,不能超过
    return Renderer(model, height=480, width=480)


def setup_camera() -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[0] = 0.0
    cam.lookat[1] = 0.0
    cam.lookat[2] = 0.4
    cam.distance = 2.2
    cam.elevation = -15.0
    cam.azimuth = 135.0
    return cam


def render_one(model, data, renderer, cam, label, hint, joint_dict) -> Path:
    apply_pose(model, data, joint_dict)
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=cam)
    img = renderer.render()
    out_path = OUT_DIR / f"{label}.png"
    from PIL import Image
    Image.fromarray(img).save(out_path)
    return out_path


def write_index(entries) -> None:
    """写一个文本索引,方便用户在文件管理器里按顺序翻看。"""
    txt = OUT_DIR / "_index.txt"
    lines = ["# 校准图集索引(按文件名前缀顺序看)", ""]
    for label, hint, path in entries:
        lines.append(f"{label}.png")
        lines.append(f"  问题: {hint}")
        lines.append(f"  文件: {path.relative_to(ROOT)}")
        lines.append("")
    txt.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", help="只渲染指定 label 的图(例如 A0)")
    args = parser.parse_args()

    if not MODEL.exists():
        print(f"模型未找到: {MODEL}", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"加载模型: {MODEL}")
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    print(f"  njnt={model.njnt}, nu={model.nu}, nq={model.nq}")
    print(f"  qpos0[0:7]={model.qpos0[:7]}  (注:无 free joint,前 7 个就是左腿的 6 个关节 + 右髋 pitch)")

    cam = setup_camera()
    renderer = make_renderer(model)

    targets = TEST_POSES
    if args.label:
        targets = [p for p in TEST_POSES if p[0] == args.label]
        if not targets:
            print(f"没找到 label={args.label}", file=sys.stderr)
            return 1

    print(f"\n开始渲染 {len(targets)} 张图到 {OUT_DIR}")
    entries = []
    for label, hint, joint_dict in targets:
        try:
            out_path = render_one(model, data, renderer, cam, label, hint, joint_dict)
            print(f"  [OK] {label:24s} -> {out_path.name}")
            entries.append((label, hint, out_path))
        except Exception as e:
            print(f"  [FAIL] {label:24s}: {e}", file=sys.stderr)

    renderer.close()
    write_index(entries)
    print(f"\n完成。共 {len(entries)}/{len(targets)} 张图。")
    print(f"索引文件: {OUT_DIR / '_index.txt'}")
    print(f"\n请按文件名顺序打开 PNG,按每张图名里的问题描述实际看到的样子,回我。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
