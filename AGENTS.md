# AGENTS.md

## 语言与沟通
- 本仓库默认使用中文沟通。
- 回答优先给可执行步骤，其次再解释原因。

## 仓库目标（当前版本）
- 目标是搭建挖掘机最小闭环：`仿真场景 -> ROS2 话题采集 -> 轨迹训练 -> 离线部署评估`。
- 代码入口：
  - 仿真：`python -m excavator_sim.run_sim`
  - 采集：`python -m excavator_sim.ros.record_topics`
  - 遥操作：`python -m excavator_sim.teleop --mode position`
  - 训练：`python -m excavator_policy.train`
  - 部署评估：`python -m excavator_policy.deploy_sim --checkpoint ... --eval-run ...`

## 环境前置约束
- 推荐在 `conda` 环境 `isaac` 中运行。
- 需要 Isaac Sim / Isaac Lab 运行时，以及 ROS2（建议 Jazzy, Ubuntu 24.04）。
- 先执行：
  - `source /opt/ros/jazzy/setup.bash`
  - `source scripts/env.sh`
- Python 依赖至少包括：
  - `numpy pandas pyarrow torch`
  - `rclpy sensor_msgs_py`

## 协作执行顺序（默认）
1. 先确认环境：Python、ROS2、Isaac 扩展可用。
2. 再确认数据链路：`/excavator/camera_front/rgb`、`/excavator/lidar/points`、`/excavator/joint_states`、`/excavator/cmd_joint`。
3. 然后做训练与评估。
4. 所有修改优先保持最小改动并可回滚。

## 关键已知问题（当前仓库）
- 默认资产路径是 `assets/excavator.usd`，但仓库里只有 URDF；需要补 USD 资产或导入流程。
- 遥操作包含 `track_joint`，与现有 4DOF URDF 关节定义不一致。
- ROS2 bridge 目前只做了基础图节点，传感器/关节完整桥接需要补齐。

## 验收标准（最小）
- 能稳定启动仿真并持续运行。
- 能录到至少一条 `run_XXX`（包含 `meta.json`、`timestamps.parquet`、`action.parquet`）。
- 训练能产出 `runs/<run_id>/model.pt`。
- 部署评估能产出 `deploy_metrics.json`。

