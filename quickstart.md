# Excavator Quickstart

```bash
conda activate isaac
```

项目默认使用 Isaac Sim 自带的 ROS2 Python 运行时。  
只有在你需要系统 `ros2 topic ...` 命令行工具时，才额外执行：

```bash
source /opt/ros/jazzy/setup.bash
```

## 1. 启动仿真

```bash
./scripts/sim.sh
```

## 2. 启动录制器

```bash
./scripts/record.sh
```

## 3. 启动 teleop

```bash
./scripts/teleop.sh
```

## 4. 可视化

```bash
./scripts/vis.sh
```

## 5. 录制控制

- `1 / A`: 开始录制
- `2 / B`: 结束录制
- `3 / X`: reset env
- `4 / Y`: joint target 归零

## 6. 检查数据

```bash
./scripts/check.sh
```

## 7. 对齐数据

```bash
./scripts/align.sh
```

## 8. 回放

```bash
./scripts/replay.sh --type raw --run-dir run_008
./scripts/replay.sh --type aligned --run-dir run_008
```

## 9. 训练

```bash
python -m excavator_policy.train --config src/excavator_policy/config.yaml
```

训练输出在：

- `logs/<run_id>/model_last.pt`
- `logs/<run_id>/model_best.pt`
- `logs/<run_id>/metrics.json`

更完整说明见 `README.md`。
