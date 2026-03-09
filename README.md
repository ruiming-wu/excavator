# Excavator Simulation, Collection, Train and Deploy project

## environment preparation
conda create -n isaac python=3.11 -y
conda activate isaac
pip install --upgrade pip
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
pip install pygame==2.6.1

## simulator
推荐使用启动脚本：

```bash
scripts/sim.sh
```

在脚本终端按 `q` 可停止仿真进程。

## teleoperation
控制与可视化已拆分为两个进程，避免界面影响操控流畅性。

控制（手柄 -> `/excavator/cmd_joint`）：

```bash
scripts/teleop.sh
```

可视化（订阅双相机、关节、ready 状态）：

```bash
scripts/vis.sh
```
