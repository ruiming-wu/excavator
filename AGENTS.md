# AGENTS.md

## 语言与沟通
- 本仓库默认使用中文沟通。
- 回答优先给可执行步骤，其次再解释原因。

## 仓库目标（当前版本）
- 目标是搭建挖掘机最小闭环：`仿真场景 -> ROS2 话题采集 -> 轨迹训练 -> 在线仿真评估`。
- 代码入口：
  - 仿真：`./scripts/sim.sh`
  - 采集：`python -m excavator_sim.record`
  - 遥操作：`python -m excavator_sim.teleop --mode position`
  - 训练：`./scripts/train.sh`
  - 评估：`./scripts/eval.sh --checkpoint ...`

## 环境前置约束
- 推荐在 `conda` 环境 `isaac` 中运行。
- 需要 Isaac Sim / Isaac Lab 运行时。
- 项目运行 `sim/teleop/record/vis` 默认使用 Isaac Sim 内置的 ROS2 Python 运行时，不再强制依赖本地 `/opt/ros/...` Python 包。
- 如果需要系统 `ros2 topic ...` 等 CLI 工具，可选安装 ROS2（建议 Jazzy, Ubuntu 24.04）并手动 `source /opt/ros/jazzy/setup.bash`。
- 先执行：
  - `source scripts/sim.sh`
- Python 依赖至少包括：
  - `numpy pandas pyarrow torch torchvision`
  - `pygame pyyaml wandb`
  - `rclpy sensor_msgs_py`（由 Isaac Sim / ROS2 环境提供）

## 协作执行顺序（默认）
1. 先确认环境：Python、Isaac 扩展可用；如果要用系统 `ros2` CLI，再确认本地 ROS2 可用。
2. 再确认数据链路：`/excavator/camera_driver/rgb`、`/excavator/camera_bucket/rgb`、`/excavator/lidar/points`、`/excavator/joint_states`、`/excavator/cmd_joint`。
3. 然后做训练与评估。
4. 所有修改优先保持最小改动并可回滚。

## 关键已知问题（当前仓库）
- 默认资产路径是 `assets/excavator.usd`，但仓库里只有 URDF；需要补 USD 资产或导入流程。
- 遥操作包含 `track_joint`，与现有 4DOF URDF 关节定义不一致。
- ROS2 bridge 目前只做了基础图节点，传感器/关节完整桥接需要补齐。
- 在线 eval 目前已经能连接真实 sim 并人工标注 episode，但“模型输出如何安全解码成在线 joint command”仍在继续调稳。
- 旧小模型 baseline 仍然很重要；当前评估脚本需要兼容大模型和小模型 checkpoint。

## 验收标准（最小）
- 能稳定启动仿真并持续运行。
- 能录到至少一条 `run_XXX`（包含 `meta.json`、`action.parquet`、`proprio.parquet` 以及相机/点云原始数据）。
- 训练能产出 `logs/<run_id>/model_last.pt`，并最好同时产出 `logs/<run_id>/model_best.pt`。
- 在线仿真评估能产出 `eval_metrics.json`。
