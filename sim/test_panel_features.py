# -*- coding: utf-8 -*-
"""Headless test for panel.py features."""
from __future__ import annotations
import json, os, sys, tempfile
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mujoco
from PyQt5.QtWidgets import QApplication, QInputDialog

import panel as P

MODEL = Path(r"D:\Code\Other\0809rebot\model\LingLong2.0\LingLong2.0.urdf")


def _set(ctl, name, v):
    slider, _ = ctl.sliders[name]
    _, lo, hi = ctl.joint_info[name]
    slider.blockSignals(True)
    slider.setValue(ctl.val_to_slider(v, lo, hi))
    slider.blockSignals(False)


def _get(ctl, name):
    slider, _ = ctl.sliders[name]
    _, lo, hi = ctl.joint_info[name]
    return ctl.slider_to_val(slider.value(), lo, hi)


def test_save_load_pose(ctl, tmp):
    out = tmp / "pose.json"
    ctl.save_pose(out)
    assert out.exists(), "save_pose did not create the file"
    snap = json.loads(out.read_text(encoding="utf-8"))
    assert len(snap) == len(ctl.sliders)
    assert "__base_z__" not in snap
    ctl.load_pose(out)
    print(f"  [OK] save/load roundtrip: {len(snap)} 关节")


def test_mirror_pose(ctl):
    """Mirror copies left_* -> right_* (and keeps left as is).

    Semantics: 'make the right side do what the left side is doing' — the user
    can then adjust the left side independently. So we don't assert left=0 after
    mirror; we assert right got left's value with the sign rule.
    """
    # Start: left = prayer pose, right = zeros
    _set(ctl, "left_shoulder_pitch_joint", -0.789); _set(ctl, "right_shoulder_pitch_joint", 0.0)
    _set(ctl, "left_shoulder_roll_joint", -0.006);  _set(ctl, "right_shoulder_roll_joint", 0.0)
    _set(ctl, "left_shoulder_yaw_joint", -0.997);   _set(ctl, "right_shoulder_yaw_joint", 0.0)
    _set(ctl, "left_elbow_joint", -0.324);          _set(ctl, "right_elbow_joint", 0.0)

    ctl.mirror_current_pose()

    # Left should be unchanged
    assert abs(_get(ctl, "left_shoulder_pitch_joint") - (-0.789)) < 1e-3
    assert abs(_get(ctl, "left_shoulder_roll_joint") - (-0.006)) < 1e-3
    assert abs(_get(ctl, "left_shoulder_yaw_joint") - (-0.997)) < 1e-3
    assert abs(_get(ctl, "left_elbow_joint") - (-0.324)) < 1e-3
    # Right should now mirror left, with sign rule:
    #   pitch / elbow  -> same sign
    #   roll / yaw     -> opposite sign
    assert abs(_get(ctl, "right_shoulder_pitch_joint") - (-0.789)) < 1e-3
    assert abs(_get(ctl, "right_shoulder_roll_joint") - 0.006) < 1e-3
    assert abs(_get(ctl, "right_shoulder_yaw_joint") - 0.997) < 1e-3
    assert abs(_get(ctl, "right_elbow_joint") - (-0.324)) < 1e-3
    print("  [OK] mirror pose: left copied to right (pitch/elbow=same, roll/yaw=opp)")


def test_mirror_motion(ctl):
    mid = next(iter(P.MOTIONS))
    m = P.MOTIONS[mid]
    nm = P.mirror_motion(m, new_id=mid+chr(0x5957), new_name=m.name+chr(0xb7)+chr(0x955c)+chr(0x50cf))
    assert nm and nm.name.endswith("\u00b7\u955c\u50cf")
    assert len(nm.keyframes) == len(m.keyframes)
    print(f"  [OK] mirror motion: {mid} -> {nm.name} ({len(nm.keyframes)} \u5e27)")


def test_t_scrub(ctl):
    mid = next(iter(P.MOTIONS))
    ctl.motion_combo.setCurrentIndex(ctl.motion_combo.findData(mid))
    m = P.MOTIONS[mid]
    ctl.t_scrub_slider.setValue(0)
    # on_t_scrub 是 build_ui 里的闭包,通过 widget signal 触发
    from PyQt5.QtCore import QEvent
    from PyQt5.QtGui import QMouseEvent
    ctl.t_scrub_slider.valueChanged.emit(0)
    for n, v in m.keyframes[0].pose.items():
        if n in ctl.sliders:
            assert abs(_get(ctl, n) - v) < 0.05
    last_t = m.keyframes[-1].t
    ctl.t_scrub_slider.setValue(int(last_t * 100))
    ctl.t_scrub_slider.valueChanged.emit(int(last_t * 100))
    print(f"  [OK] t scrubber t=0..{last_t:.2f} no crash, matches first kf")


def test_recording(ctl):
    ctl.start_recording()
    assert ctl._is_recording
    _set(ctl, "head_pitch_joint", 0.3)
    ctl.on_tick_recording()
    ctl.on_tick_recording()
    ctl.on_tick_recording()
    assert len(ctl._record_buffer) == 1
    _set(ctl, "head_pitch_joint", 0.6)
    ctl.on_tick_recording()
    assert len(ctl._record_buffer) == 2
    with mock.patch.object(QInputDialog, "getText", return_value=("test_motion", True)):
        ctl.stop_recording()
    assert not ctl._is_recording
    print("  [OK] recording dedup: 3 same-tick + 1 changed = 2 \u5e27")


def main():
    if not MODEL.exists():
        print(f"\u6a21\u578b\u672a\u627e\u5230: {MODEL}", file=sys.stderr)
        return 2
    app = QApplication.instance() or QApplication(sys.argv)
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)
    ctl = P.Controller(model, data)
    saved = P.load_all_saved_motions()
    if saved:
        P.MOTIONS.update(saved)
    from PyQt5.QtWidgets import QMainWindow
    window = QMainWindow()
    P.build_ui(window, ctl)
    try:
        with tempfile.TemporaryDirectory() as td:
            print("\n=== test_save_load_pose ===")
            test_save_load_pose(ctl, Path(td))
        print("\n=== test_mirror_pose ==="); test_mirror_pose(ctl)
        print("\n=== test_mirror_motion ==="); test_mirror_motion(ctl)
        print("\n=== test_t_scrub ==="); test_t_scrub(ctl)
        print("\n=== test_recording ==="); test_recording(ctl)
        print("\nAll tests passed.")
        return 0
    except AssertionError as e:
        print(f"\n[FAIL] {e}", file=sys.stderr)
        import traceback; traceback.print_exc(); return 1
    except Exception as e:
        print(f"\n[FAIL] {type(e).__name__}: {e}", file=sys.stderr)
        import traceback; traceback.print_exc(); return 1


if __name__ == "__main__":
    raise SystemExit(main())
