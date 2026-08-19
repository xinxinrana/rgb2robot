# 005. 路线 A MVP 跑通记录

> 2026-08-09 第一次端到端跑通 `mp4 → MediaPipe → 几何 IK → Motion JSON → panel.py ▶ 播放`。
> 实现按 `004` 文档里 A0~A2 拆。

---

## 1. 跑通事实

| 项 | 值 |
|---|---|
| 输入视频 | `D:\Downloads\RGB视频数据集.mp4`(720x1280, 22 秒, 30fps, 661 帧) |
| MediaPipe 模型 | `tools/models/pose_landmarker.task` (5.7MB lite 版, Tasks API) |
| 关键点 JSON | `out/in_video.pose.json` (4.5MB, 661 帧, 33 关键点 × {image + world}) |
| 机器人 Motion JSON | `sim/saved_motions/in_video.json` (118KB, 221 关键帧, 10fps) |
| 端到端验证 | panel.py 启动加载成功,`apply_motion` 无 NaN,qpos 在限位内 |
| 处理耗时 | 视频解码 + MediaPipe 推理 ~数秒;scipy 平滑 + 降采样 < 1 秒 |

`panel.py` 启动后下拉里会多出 **`in_video`** 动作,选它 ▶ 就能让机器人演这 22 秒。

---

## 2. 文件清单

```
tools/
  models/
    pose_landmarker.task            # MediaPipe 模型,git 不上传(可脚本下载)
  pose_extract.py                   # A0: mp4 → pose.json
  retarget.py                       # A1+A2: pose.json → Motion JSON(几何 IK + 平滑 + 降采样)
  make_synthetic_pose.py            # 合成 pose.json(回归测试,无视频依赖)
  make_test_video.py                # 画一段挥手火柴人视频(回归测试)
out/
  in_video.pose.json                # 真实视频的 MediaPipe 关键点
  _test_wave.pose.json              # 合成数据(测试 pipeline 用)
  _test_wave.mp4                    # 合成视频(测试用)
sim/saved_motions/
  in_video.json                     # 真实视频的机器人动作(panel.py 可播)
in_video.mp4                        # 输入视频(不传 git,本地放)
```

---

## 3. 用法

```bash
# 一次性装依赖
pip install mediapipe opencv-python scipy

# 下载 MediaPipe 模型(只第一次需要)
python -c "import urllib.request; urllib.request.urlretrieve(
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task',
  'tools/models/pose_landmarker.task')"

# 跑 A0(视频→关键点)
python tools/pose_extract.py in_video.mp4 -o out

# 跑 A1+A2(关键点→Motion JSON, 默认 10fps)
python tools/retarget.py out/in_video.pose.json --name in_video

# 启动 panel.py 验证
cd sim && python panel.py
# 下拉选 in_video,按 ▶
```

---

## 4. MVP 设计取舍

按"先跑通,不求完美"原则:

- **只做上半身**:head_yaw/pitch, waist_yaw/pitch, 双臂 pitch/roll/elbow(10 个关节)。其它 20 个锁 0。
- **wrist 三轴 + 双腿 + 肩 yaw 都不算**:MVP 阶段不重要,真要算就当 IK 输出目标点时让用户手动覆盖。
- **几何 IK 极简**:肩 pitch/roll 直接用 3D 向量点积 + atan2 算;肘角用上臂-小臂夹角线性映射。**没做完整 IK,所有关节角都从骨架向量反解,精度有限**。
- **MediaPipe Tasks API**(`mp_pose` 已废,新版用 `PoseLandmarker` + `.task` 模型文件)。
- **MediaPipe 单目 3D 已知不精确**:侧身、遮挡时 Z 跳;head_yaw 在 t=5s 打到上限 +1.57 就是这个原因。后续可换 RTMPose 缓解。
- **窗口=11 的 Savitzky-Golay 平滑** + **30→10fps 降采样**(3:1)。

---

## 5. 已知问题(下一步要修)

| # | 问题 | 触发条件 | 修法候选 |
|---|---|---|---|
| 1 | 头/肩 yaw 容易撞限位 | 视频里人侧身时 MediaPipe 把"侧身"误判成"头大幅偏航" | 改用 RTMPose;或对 yaw 角加更激进平滑;或限制 yaw 变化率 |
| 2 | 坐姿时 `shoulder_pitch=+2.97`(打满) | 视频 f0 的人手在腿上,世界 Y 是"上",肩→肘向量偏 +X(人的身体横向放),被 IK 当成"大臂后摆" | 校准时先做一次"归零"标定:把第一帧的躯干姿态当作世界 +Y(忽略 MediaPipe 自带世界坐标) |
| 3 | 左半边常 visibility<0.3 | 22 秒里人多次侧身,左臂在画面外 | 简单做法:visibility<0.3 的帧用上一帧的关节角"糊"过去;或对整个左臂加缺失检测 |
| 4 | 肘角映射 `(π/2 - ang) / (π/2)` 是经验公式 | 真实合十/推掌动作肘角在 (-0.3, 0.5) 区间时还凑合,大幅弯(>90°)会失真 | 等真实视频多了再校准;或直接读 MuJoCo 关节限位反推 |
| 5 | 没做 unit test | —— | 至少给 `_retarget_one` 写一组向量→角度的 sanity case(比如"手垂下"→pitch=0,roll=0) |
| 6 | 视频里有人体移动,机器人不会"走" | 机器人没 free joint,平移靠不了 | 在 `004` 文档里已说明是约束,接受 |
| 7 | 22 秒太长(221 关键帧) | panel.py ▶ 播放 1× 速度太慢,3× 才舒服 | 后续加 5fps 降采样选项 |

---

## 6. 还没接进 panel.py 的事

按 `004` 文档,**A3** 才是新增"📥 导入视频"按钮。当前还**没做**:
- panel.py 里加按钮 → 弹文件选择 → 调 `subprocess.run(["python", "tools/pose_extract.py", ...])` + `retarget.py` → 自动 ▶ 播放

为啥先不做 A3:
- A0+A1+A2 已经跑通,验证流程手工两步(`pose_extract` + `retarget`)也行
- panel.py 加 subprocess 比较脆(路径、跨平台、错误处理),MVP 阶段先不接

要做 A3 时:
- panel.py 顶部 import `subprocess, tempfile, os`
- Controller 加 `def on_import_video(self): ...` 方法
- 走 QFileDialog.getOpenFileName,选 mp4 → 调两个脚本 → 把输出 Motion 直接塞进 `P.MOTIONS["in_video"]` 并切换播放
- 建议同时在状态栏 log 进度("抽关键点 30%..." → "反解关节角..." → "加载完成")

---

## 7. 下一步建议(等用户拍板)

| 选项 | 时间 | 价值 |
|---|---|---|
| **A3**: panel.py 加"导入视频"按钮 | 0.5 天 | 一次点击跑通,可用性 10× |
| 修问题 1+2(改用 RTMPose + 归一化) | 1 天 | 端到端效果明显变好 |
| 给 retarget.py 写 5 个 unit test | 0.3 天 | 防回归,符合"先写 test"的反馈 |
| 接 A4:录 3 段真人视频打磨 | 1~2 天 | 调参出"能拿出手"的 demo |

