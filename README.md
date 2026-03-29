# Excavator Pipeline README

## 1. 项目目标

当前仓库已经形成一条最小闭环：

1. Isaac Sim 挖掘机场景仿真
2. ROS2 话题遥操作与录制
3. 原始数据回放与对齐
4. 对齐后多模态策略训练
5. 离线评估与后续部署准备

当前默认工作流是：

`仿真 -> teleop -> record -> replay/check -> align -> train`

---

## 2. 目录结构

核心目录现在是：

- `data/raw/`: 原始录制数据，每条 run 一个目录
- `data/aligned/`: 对齐后的数据，每条 run 一个目录，核心文件是 `frames.parquet`
- `logs/`: 训练输出、checkpoint、metrics、wandb 本地日志目录
- `assets/`: URDF 与场景资源
- `src/excavator_sim/`: 仿真、teleop、record、vis
- `src/excavator_policy/`: 训练数据集、模型、训练与部署脚本
- `scripts/`: 所有统一启动脚本

---

## 3. 环境准备

推荐环境：

- Ubuntu 24.04
- `conda` 环境：`isaac`
- Isaac Sim 5.1
- 本地安装 ROS2 CLI 可选；项目运行时默认使用 Isaac Sim 自带的 ROS2 Python 运行时

### 3.1 激活环境

```bash
conda activate isaac
```

### 3.2 Python 依赖

建议至少包含：

```bash
pip install numpy pandas pyarrow fastparquet pygame open3d pyyaml wandb
```

训练相关依赖：

```bash
pip install torch torchvision
```

说明：

- `pyarrow` / `fastparquet`: 用于 parquet 读写
- `PyYAML`: 用于读取 `src/excavator_policy/config.yaml`
- `wandb`: 训练日志记录
- `torchvision`: `ResNet-50` 图像 encoder
- `rclpy` / `sensor_msgs_py` 运行时由 Isaac Sim 内置 `isaacsim.ros2.bridge` 提供
- 如果你只想运行本仓库脚本，不再要求先 `source /opt/ros/jazzy/setup.bash`
- 只有在你需要系统 `ros2 topic ...` 命令行工具时，才额外 `source /opt/ros/jazzy/setup.bash`

---

## 4. 启动仿真

启动：

```bash
./scripts/sim.sh
```

停止：

- 在该终端按 `q`
- 或 `Ctrl+C`

### 4.1 当前仿真设置

当前场景包含：

- 4DOF 挖掘机（URDF 导入）
- 卡车
- 土堆 / stone pile
- 安装在 excavator house 的前向相机
- 安装在 excavator bucket 的相机
- 一个 lidar
- ROS2 bridge

### 4.2 当前传感器设置

相机：

- `camera_driver`
- `camera_bucket`
- 分辨率：`240 x 160`
- 录制存储 shape：`(160, 240, 3)`

lidar：

- 录制存储 shape：`(N, 3)`
- 单帧点数不是常数
- 当前训练前会随机下采样到 `4096` 点

### 4.3 当前主 ROS2 话题

仿真侧主要话题：

- `/excavator/camera_driver/rgb`
- `/excavator/camera_bucket/rgb`
- `/excavator/lidar/points`
- `/excavator/joint_states`
- `/excavator/cmd_joint`
- `/excavator/reset`
- `/excavator/ready`
- `/excavator/stones_in_truck`
- `/excavator/episode_meta`
- `/excavator/record_control`

---

## 5. Teleop

启动 teleop：

```bash
./scripts/teleop.sh
```

当前 teleop 是纯控制进程，不做重 UI 渲染，目的是保证操控顺滑。

### 5.1 手柄控制逻辑

当前主要映射：

- 左摇杆上下：`arm_joint`
- 左摇杆左右：`swing_joint`
- 右摇杆上下：`boom_joint`
- 右摇杆左右：`bucket_joint`

控制模式本质上是：

`target_joint = current_target_joint + scale * joystick_value`

### 5.2 特殊按键

- `A / 键盘 1`: 开始录制
- `B / 键盘 2`: 结束录制
- `X / 键盘 3`: reset env
- `Y / 键盘 4`: joint target 归零

---

## 6. 可视化

启动在线可视化：

```bash
./scripts/vis.sh
```

当前 `vis` 只负责显示，不参与控制和写盘。

### 6.1 当前可视化内容

- 两路相机图像
- lidar 视图
- 当前 joint state
- 当前 target joint command
- ready 状态
- 操作提示

### 6.2 lidar 视角

当前 lidar 可视化视角：

- 观察点：`(0, 0, 0)` 或在 replay 中保持统一指定视角
- 颜色按距离映射

---

## 7. 录制

启动录制器：

```bash
./scripts/record.sh
```

录制器是独立进程，teleop 只发录制控制信号，真正的写盘由 recorder 完成。

### 7.1 每条 raw run 会保存什么

每条 `data/raw/run_xxx/` 当前包含：

- `meta.json`
- `camera_driver.parquet`
- `camera_bucket.parquet`
- `lidar.parquet`
- `proprio.parquet`
- `action.parquet`
- `stones_in_truck.parquet`
- `record_control.parquet`
- `camera_driver/*.npy`
- `camera_bucket/*.npy`
- `lidar/*.npy`

### 7.2 已录内容含义

- `camera_driver/*.npy`: driver 相机图像
- `camera_bucket/*.npy`: bucket 相机图像
- `lidar/*.npy`: 原始点云 `(N, 3)`
- `proprio.parquet`: 实际关节状态（位置、速度）
- `action.parquet`: teleop 发出的关节命令
- `stones_in_truck.parquet`: 卡车中 stone 数量
- `meta.json`: episode 元数据、平均频率、时长等

### 7.3 为什么同时保留 `proprio` 和 `action`

这是当前设计里很重要的一点：

- `proprio`: 机器人实际状态
- `action`: 操作者命令

它们不应该混成一个东西，因为命令不一定被物理系统完全实现。

---

## 8. 原始回放与对齐回放

启动 replay：

```bash
./scripts/replay.sh --run-dir run_008
```

### 8.1 raw 模式

```bash
./scripts/replay.sh --type raw --run-dir run_008
```

按原始 `recv_ns` 回放：

- 两路相机
- lidar
- current joints
- target joints
- stones in truck

### 8.2 aligned 模式

```bash
./scripts/replay.sh --type aligned --run-dir run_008
```

按 `data/aligned/run_xxx/frames.parquet` 回放。

### 8.3 replay 里期望看到什么

正常情况下你应该看到：

- 双相机画面随挖掘机动作变化
- lidar 点云同步更新
- `cur` 与 `tgt` 的关节值随着操作变化
- `stones in truck` 随装载逐渐变化
- aligned 模式下整体节奏更规整

---

## 9. 数据检查

运行：

```bash
./scripts/check.sh
```

当前检查规则包括：

- 初始 joint state 是否在 `±0.1 rad` 内
- 起始 `stones_in_truck == 0`
- 结束 `stones_in_truck >= 8% * 总石块数`
- episode 时长是否在 `[20s, 50s]`
- 相机和 lidar 平均频率是否 `> 15 Hz`

并且会输出通过 run 的统计摘要。

---

## 10. 数据对齐

运行：

```bash
./scripts/align.sh
```

### 10.1 当前对齐规则

- 默认主轴：`lidar`
- 时间基准：`recv_ns`
- 严格 `±50ms` 窗口的模态：
  - `camera_driver`
  - `camera_bucket`
  - `lidar`
  - `proprio`
  - `action`
- `stones_in_truck`：只找最近邻，不受 `50ms` 窗口限制

### 10.2 对齐产物

每条 run 会在：

- `data/aligned/run_xxx/frames.parquet`
- `data/aligned/run_xxx/align_meta.json`

其中 `frames.parquet` 包含：

- 对齐后的图像路径
- 对齐后的点云路径
- 当前 `proprio_*`
- 当前 `action_*`
- `stones_in_truck_*`
- 各模态相对主轴的 `delta_ms_*`

---

## 11. 当前训练设置

训练入口：

```bash
python -m excavator_policy.train --config src/excavator_policy/config.yaml
```

### 11.1 当前训练数据源

训练读取：

- `data/aligned/run_xxx/frames.parquet`
- 对应 raw run 中的图像 / 点云文件

### 11.2 当前 observation / target

当前 observation：

- `camera_driver`: `(3, 160, 240)`
- `camera_bucket`: `(3, 160, 240)`
- `points`: `(4096, 3)`
- `current_state`: `(4,)`

当前 target：

- `future_cmd`: `(16, 4)`

也就是：

- 输入：当前双相机 + 当前点云 + 当前真实 joint state
- 输出：未来 16 步 joint command 序列

### 11.3 当前 joint 顺序

训练侧统一顺序为：

- `swing_joint`
- `boom_joint`
- `arm_joint`
- `bucket_joint`

这一步会把原始数据里不一致的 `name` 顺序重排成统一格式。

---

## 12. 当前模型结构

文件：

- [src/excavator_policy/model.py](src/excavator_policy/model.py)

当前模型是一个紧凑版 diffusion-style policy：

1. `camera_driver` 编码
2. `camera_bucket` 编码
3. 点云 MLP + mean pooling 编码
4. 当前 command 编码
5. 条件特征融合
6. 对未来 `16 x 4` 命令序列进行 denoise 预测

### 12.1 结构概览

- 图像编码器：共享权重的 `ResNet-50`
- 点云编码器：PointNet 风格编码器（逐点 MLP + max pooling）
- 状态编码器：2 层 MLP
- 条件向量：直接拼接成 `2048` 维
- denoiser：MLP

### 12.2 当前不是的东西

当前模型还不是一个大规模 transformer policy，也不是 PointNet++ / 3D sparse conv 模型。它现在是一个正式一些的多模态 diffusion-style baseline：`ResNet-50 + PointNet + state MLP + MLP denoiser`。

---

## 13. 训练输出与日志

现在训练输出写到：

- `logs/<run_id>/model_last.pt`
- `logs/<run_id>/model_best.pt`
- `logs/<run_id>/metrics.json`
- `logs/<run_id>/config.json`
- `logs/<run_id>/history.json`
- `logs/<run_id>/split.json`

说明：

- `model_last.pt`: 最后一个 epoch 的 checkpoint
- `model_best.pt`: 验证集最优 checkpoint
- `metrics.json` 中会记录 `best_val_epoch`

### 13.1 wandb

训练现在会使用 `wandb`。

默认配置在：

- [src/excavator_policy/config.yaml](src/excavator_policy/config.yaml)

当前默认：

- `project: excavator_policy`
- `name: excavator_policy`

如果当前环境没有 `wandb`，训练会直接报错提醒安装。

---

## 14. 当前最推荐的完整工作流

### Step 1: 启动仿真

```bash
./scripts/sim.sh
```

### Step 2: 启动 recorder

```bash
./scripts/record.sh
```

### Step 3: 启动 teleop

```bash
./scripts/teleop.sh
```

### Step 4: 可选启动 vis

```bash
./scripts/vis.sh
```

### Step 5: 采集多条 raw run

- `1/A` 开始
- `2/B` 结束
- `3/X` reset env
- `4/Y` joint target 归零

### Step 6: 检查数据

```bash
./scripts/check.sh
```

### Step 7: 对齐数据

```bash
./scripts/align.sh
```

### Step 8: 回放 raw / aligned

```bash
./scripts/replay.sh --type raw --run-dir run_008
./scripts/replay.sh --type aligned --run-dir run_008
```

### Step 9: 训练

```bash
python -m excavator_policy.train --config src/excavator_policy/config.yaml
```

---

## 15. 当前状态总结

现在仓库已经具备：

- 可运行的 Isaac Sim 挖掘机场景
- 稳定的 teleop 与 recorder 链路
- raw 数据与 aligned 数据两层表示
- replay / check / align 工具链
- 可训练的多模态 diffusion-style baseline
- wandb + 本地 `logs/` 训练日志落盘

如果下一步继续推进，最值得做的是：

1. run-level train/val split
2. 更强的点云编码器
3. 更真实的 diffusion sampling / inference 过程
4. sim 内在线部署验证闭环
