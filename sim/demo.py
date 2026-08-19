"""驱动灵龙2.0 在 MuJoCo viewer 里动起来。

策略:用 mj_forward(运动学)逐帧覆盖 qpos,让双臂/头部按时间正弦摆动。
- 不跑动力学(不调 mj_step),所以机器人不会因为重力摔倒。
- 想看"物理仿真"(重力、接触、行走)需要 actuator + 控制器,那是另一码事。
"""
from pathlib import Path
import time
import numpy as np
import mujoco
import mujoco.viewer

MODEL = Path(__file__).resolve().parent.parent / "model" / "LingLong2.0" / "LingLong2.0.urdf"


def resolve_joint_addr(model: mujoco.MjModel, name: str) -> int:
    """返回关节在 qpos 里的下标;找不到就报错并列出全部关节名。"""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        names = [model.joint(i).name for i in range(model.njnt)]
        raise SystemExit(f"未找到关节 '{name}'。全部关节:\n  " + "\n  ".join(names))
    return model.joint(jid).qposadr[0]


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)

    # 要动的关节:双臂挥动 + 头部摆头
    joints = {
        "right_shoulder_pitch_joint": resolve_joint_addr(model, "right_shoulder_pitch_joint"),
        "right_shoulder_roll_joint":  resolve_joint_addr(model, "right_shoulder_roll_joint"),
        "right_elbow_joint":          resolve_joint_addr(model, "right_elbow_joint"),
        "left_shoulder_pitch_joint":  resolve_joint_addr(model, "left_shoulder_pitch_joint"),
        "left_shoulder_roll_joint":   resolve_joint_addr(model, "left_shoulder_roll_joint"),
        "left_elbow_joint":           resolve_joint_addr(model, "left_elbow_joint"),
        "head_yaw_joint":             resolve_joint_addr(model, "head_yaw_joint"),
    }
    for name, addr in joints.items():
        print(f"  {name:32s} -> qpos[{addr}]")

    print("\n正在启动 viewer... 关闭窗口或按 Ctrl+C 退出。")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mj_resetData(model, data)
        t0 = time.time()
        while viewer.is_running():
            t = time.time() - t0
            # 右手:前后摆动 + 抬起 + 肘弯
            data.qpos[joints["right_shoulder_pitch_joint"]] = -0.9 + 0.4 * np.sin(t * 2.0)
            data.qpos[joints["right_shoulder_roll_joint"]]  =  0.5
            data.qpos[joints["right_elbow_joint"]]          =  1.2
            # 左手:反相摆动
            data.qpos[joints["left_shoulder_pitch_joint"]]  = -0.9 - 0.4 * np.sin(t * 2.0)
            data.qpos[joints["left_shoulder_roll_joint"]]   = -0.5
            data.qpos[joints["left_elbow_joint"]]           =  1.2
            # 头部左右慢慢转
            data.qpos[joints["head_yaw_joint"]]             =  0.4 * np.sin(t * 0.7)

            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.02)  # ~50 Hz


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已退出。")
