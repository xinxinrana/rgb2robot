# -*- coding: utf-8 -*-
"""A0 测试视频生成:在画面里画一个挥手的"火柴人"。

不依赖真人素材,只验证 pipeline(pose_extract → retarget → motion)能跑通。
输出 out/_test_wave.mp4
"""
from __future__ import annotations
import math, os
import cv2
import numpy as np

OUT = "out/_test_wave.mp4"
W, H = 640, 480
FPS = 30
DUR = 4.0  # 秒
N = int(FPS * DUR)


def lerp(a, b, t):  return a + (b - a) * t
def lerp_pt(p, q, t): return (int(lerp(p[0], q[0], t)), int(lerp(p[1], q[1], t)))


def wave_frame(t: float) -> np.ndarray:
    img = np.full((H, W, 3), 230, dtype=np.uint8)
    cv2.line(img, (40, 380), (W - 40, 380), (120, 120, 120), 1)

    cx = W // 2
    head_y = 130
    cv2.circle(img, (cx, head_y), 24, (60, 60, 60), 2)
    neck = (cx, head_y + 24)
    sway = 8 * math.sin(2 * math.pi * t * 0.5)
    hip = (cx + int(sway), 280)
    cv2.line(img, neck, hip, (60, 60, 60), 3)

    l_shoulder = (cx - 30, neck[1] + 4)
    l_elbow = (cx - 50, neck[1] + 60)
    l_hand = (cx - 55, neck[1] + 110)
    for a, b in [(l_shoulder, l_elbow), (l_elbow, l_hand)]:
        cv2.line(img, a, b, (60, 60, 60), 3)
    cv2.circle(img, l_hand, 5, (60, 60, 60), -1)

    phase = (t / DUR) * 2 * math.pi
    raise_amt = 0.5 + 0.5 * math.sin(2 * math.pi * t * 0.75)
    r_shoulder = (cx + 30, neck[1] + 4)
    r_hand_target = (cx + 50, head_y - 50)
    r_hand_rest = (cx + 55, neck[1] + 110)
    r_hand = lerp_pt(r_hand_rest, r_hand_target, raise_amt)
    r_elbow = lerp_pt(
        (cx + 50, neck[1] + 60),
        (cx + 50, head_y + 5),
        raise_amt,
    )
    for a, b in [(r_shoulder, r_elbow), (r_elbow, r_hand)]:
        cv2.line(img, a, b, (60, 60, 60), 3)
    cv2.circle(img, r_hand, 5, (60, 60, 60), -1)

    l_hip = (cx - 15, hip[1])
    r_hip = (cx + 15, hip[1])
    l_foot = (cx - 25, 380)
    r_foot = (cx + 25, 380)
    for a, b in [(l_hip, l_foot), (r_hip, r_foot)]:
        cv2.line(img, a, b, (60, 60, 60), 3)
    cv2.circle(img, l_foot, 5, (60, 60, 60), -1)
    cv2.circle(img, r_foot, 5, (60, 60, 60), -1)

    cv2.putText(img, f"t={t:.2f}s", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return img


def main():
    os.makedirs("out", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(OUT, fourcc, FPS, (W, H))
    for i in range(N):
        t = i / FPS
        vw.write(wave_frame(t))
    vw.release()
    print(f"[OK] {OUT}  {N} 帧 @ {FPS} fps")


if __name__ == "__main__":
    main()
