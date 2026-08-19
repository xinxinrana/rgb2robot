"""打印灵龙2.0 的全部关节清单。

加载模型后遍历 njnt,输出 ID / 名称 / 类型 / 限位。
运行:python list_joints.py
"""
from pathlib import Path
import mujoco

MODEL = Path(__file__).resolve().parent.parent / "model" / "LingLong2.0" / "LingLong2.0.urdf"
TYPE_NAMES = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}


def main() -> None:
    if not MODEL.exists():
        raise SystemExit(f"模型文件不存在: {MODEL}")
    model = mujoco.MjModel.from_xml_path(str(MODEL))

    print("=" * 72)
    print(f"灵龙2.0 关节清单  (njnt={model.njnt})")
    print("=" * 72)
    print(f"{'ID':>3}  {'name':<32} {'type':<6} {'lower':>8} {'upper':>8}")
    print("-" * 72)
    for i in range(model.njnt):
        name = model.joint(i).name or f"<unnamed_{i}>"
        type_id = int(model.jnt_type[i])
        type_str = TYPE_NAMES.get(type_id, f"未知({type_id})")
        lo, hi = float(model.jnt_range[i][0]), float(model.jnt_range[i][1])
        print(f"{i:>3}  {name:<32} {type_str:<14} {lo:>8.3f} {hi:>8.3f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
