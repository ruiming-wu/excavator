# Excavator Pipeline

## 1. 当前目标

当前仓库的目标是跑通最小闭环：

`Isaac Sim 仿真 -> teleop 采集 -> 数据对齐 -> 策略训练 -> 在线仿真评估`

当前主要入口：

- 仿真：`./scripts/sim.sh`
- 遥操作：`./scripts/teleop.sh`
- 录制：`./scripts/record.sh`
- 回放：`./scripts/replay.sh`
- 数据检查：`./scripts/check.sh`
- 数据对齐：`./scripts/align.sh`
- 训练：`./scripts/train.sh`
- 在线评估：`./scripts/eval.sh --checkpoint logs/<run_id>/model_best.pt`
- 离线分析：`./scripts/analyse.sh --checkpoint ...`

---

## 2. 目录结构

- `assets/`: URDF 与相关资源
- `data/raw/`: 原始录制数据
- `data/aligned/`: 对齐后的数据，每条 run 对应一个 `frames.parquet`
- `logs/`: 训练输出、评估输出、分析报告
- `src/excavator_sim/`: 仿真、teleop、record、vis
- `src/excavator_policy/`: 数据集、模型、训练、评估、分析
- `scripts/`: 统一脚本入口

---

## 3. 环境准备

### 3.1 创建 conda 环境

```bash
conda create -n excavator_env python=3.11 -y
conda activate excavator_env
```

### 3.2 安装依赖

```bash
pip install --upgrade pip
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install pyarrow==23.0.1 fastparquet==2025.12.0 PyYAML==6.0.2 pygame==2.6.1
```

### 3.3 运行时环境

运行仓库脚本前，先：

```bash
conda activate excavator_env
```

然后直接运行 `./scripts/*.sh`。

### 3.4 ROS2 依赖说明

- 项目运行 `sim / teleop / record / eval / vis` 时，默认使用 Isaac Sim 自带的 ROS2 Python 运行时
- 不再强制依赖本地 `/opt/ros/...` 的 Python 包
- 如果你想用系统 `ros2 topic ...` 命令行工具，可以额外安装本地 ROS2 CLI

---

## 4. 仿真、采集与回放

### 4.1 启动仿真

```bash
./scripts/sim.sh
```

当前仿真主要话题：

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

### 4.2 遥操作

```bash
./scripts/teleop.sh
```

teleop 发送的是关节目标命令，当前录制会同时保存：

- `action`: 操作者命令
- `proprio`: 机器人真实状态

### 4.3 录制

```bash
./scripts/record.sh
```

每条 `data/raw/run_xxx/` 当前会包含：

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

### 4.4 回放

原始回放：

```bash
./scripts/replay.sh --type raw --run-dir run_008
```

对齐后回放：

```bash
./scripts/replay.sh --type aligned --run-dir run_008
```

### 4.5 数据检查

```bash
./scripts/check.sh
```

### 4.6 数据对齐

```bash
./scripts/align.sh
```

当前对齐规则：

- 主轴默认是 `lidar`
- 时间基准使用 `recv_ns`
- `camera_driver / camera_bucket / lidar / proprio / action` 使用严格 `±50ms`
- `stones_in_truck` 只做最近邻，不受 `50ms` 限制

输出到：

- `data/aligned/run_xxx/frames.parquet`
- `data/aligned/run_xxx/align_meta.json`

---

## 5. 训练

### 5.1 数据定义

训练读取：

- `data/aligned/run_xxx/frames.parquet`
- 对应 raw run 里的图像和点云 `.npy`

当前 observation：

- `camera_driver`: `(3, 160, 240)`
- `camera_bucket`: `(3, 160, 240)`
- `points`: `(4096, 3)`
- `current_state`: `(4,)`

当前 target：

- 未来 `16` 步 joint command 序列
- shape: `(16, 4)`

统一 joint 顺序：

- `swing_joint`
- `boom_joint`
- `arm_joint`
- `bucket_joint`

### 5.2 当前训练目标

当前训练使用 **flow matching** 风格。

训练路径：

- `x0 = 0`
- `x1 = future_cmd_seq`
- `x_t = t * future_cmd_seq`

监督目标：

- `velocity = future_cmd_seq`

当前 loss：

- `MSE(pred_velocity, future_cmd_seq)`

### 5.3 当前模型

当前模型定义：

- `src/excavator_policy/model.py`

训练入口：

```bash
./scripts/train.sh
```

默认配置：

- `src/excavator_policy/config.yaml`

### 5.4 当前训练输出

训练输出在：

- `logs/<run_id>/model_last.pt`
- `logs/<run_id>/model_best.pt`
- `logs/<run_id>/metrics.json`
- `logs/<run_id>/history.json`
- `logs/<run_id>/split.json`
- `logs/<run_id>/config.json`

如果使用 wandb，默认 run 名就是日志目录名：

- `YYYYmmdd_HHMMSS`

---

## 6. 在线评估

入口：

```bash
./scripts/eval.sh --checkpoint logs/<run_id>/model_best.pt
```

当前评估器是**在线人工闭环评估器**，会读取真实 sim observation，再在线发布 `/excavator/cmd_joint`。

当前交互：

- `m`: 开始/结束当前 episode
- `s`: 标记上一条 episode 成功
- `f`: 标记上一条 episode 失败
- `r`: 手动 reset 环境
- `q`: 退出评估

每次 `s / f` 后会立刻写：

- `logs/eval_<timestamp>/eval_metrics.json`

另外每隔固定推理次数还会写：

- `logs/eval_<timestamp>/inference_debug.jsonl`

用于后续分析“模型是否卡在局部姿态”。

### 6.1 当前重建过程

当前 eval 使用 flow matching 的欧拉积分重建：

- 初始动作序列：全 0
- 默认积分步数：`10`
- 默认步长：`0.1`

即：

```text
action_{k+1} = action_k + 0.1 * v_theta(obs, action_k, t_k)
```

### 6.2 当前状态

当前在线评估已经能稳定：

- 连接真实 sim
- 人工标注 episode
- 输出 report 和 debug 快照

但策略是否能稳定推进完整挖装阶段，仍在持续调试。

---

## 7. 离线分析

新增分析脚本：

```bash
./scripts/analyse.sh \
  --checkpoint logs/<run_id>/model_best.pt \
  --split val
```

它会比较：

- 真实未来 16 步动作
- 模型重建的未来 16 步动作

并输出：

- overall `MAE / RMSE`
- 第 1~16 步误差
- 真实/预测序列内部平均运动幅度
- hesitation ratio

输出到：

- `logs/analysis_<timestamp>/prediction_analysis.json`

---

## 8. 当前已知问题

- 在线 eval 虽然能跑，但策略仍可能卡在局部姿态，尚未稳定完成“挖取 -> 抬臂 -> 回转 -> 放料”完整阶段切换
- 目前正在重点排查：数据中的长犹豫片段、短 horizon、单帧 observation 带来的局部 continuation 问题
- 小模型 baseline 仍然很重要，不建议轻易删除

---

## 9. 最小可验收标准

- 能稳定启动仿真
- 能录到至少一条 `run_xxx`
- 能完成 `check` 与 `align`
- 能训练出 `logs/<run_id>/model_last.pt` 和 `model_best.pt`
- 能跑在线评估并输出 `eval_metrics.json`
