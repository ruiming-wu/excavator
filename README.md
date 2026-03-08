# Excavator Simulation, Collection, Train and Deploy project

## environment preparation
conda create -n isaac python=3.11 -y
conda activate isaac
pip install --upgrade pip
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

make sure to install opt ros2 in your system
ros2 jazzy (Ubuntu24) is recommended

## simulator
推荐使用启动脚本：

```bash
scripts/sim.sh
```

在脚本终端按 `q` 可停止仿真进程。
