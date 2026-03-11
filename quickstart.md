# Excavator Pipeline Quickstart

## 1) Load env / start sim launcher

```bash
source scripts/sim.sh
```

## 2) Run Isaac Sim minimal scene

```bash
scripts/sim.sh
```

## 3) Teleop + record one trajectory

Terminal A:
```bash
scripts/sim.sh
```

Terminal B:
```bash
scripts/record.sh
```

Terminal C:
```bash
scripts/teleop.sh
```

录制控制：
```text
1 / A -> start recording
2 / B -> finish recording and save
3 / X -> reset env
4 / Y -> zero joints
```

## 4) Collect >= 30 trajectories

```bash
for i in $(seq 1 30); do
  echo "Collect run $i"
  # keep recorder and sim running, use 1/A to start and 2/B to finish each demo
done
```

## 5) Replay validation

```bash
scripts/replay.sh --run-dir run_000
```

或者直接运行：

```bash
python data/replay.py --run-dir run_000
```

## 6) Train first policy

```bash
python -m excavator_policy.train --epochs 20 --batch-size 16
```

## 7) Deployment validation

```bash
python -m excavator_policy.deploy_sim \
  --checkpoint runs/<run_id>/model.pt \
  --eval-run data/raw/run_000
```
