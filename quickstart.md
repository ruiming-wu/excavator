# Excavator Pipeline Quickstart

## 1) Load env

```bash
source scripts/env.sh
```

## 2) Run Isaac Sim minimal scene

```bash
python -m excavator_sim.run_sim
```

## 3) Teleop + record one trajectory

Terminal A:
```bash
python -m excavator_sim.run_sim
```

Terminal B:
```bash
python -m excavator_sim.ros.record_topics
```

Terminal C:
```bash
python -m excavator_sim.teleop --mode position
```

## 4) Collect >= 30 trajectories

```bash
for i in $(seq 1 30); do
  echo "Collect run $i"
  python -m excavator_sim.ros.record_topics
  # stop with Ctrl+C after each expert demo
  sleep 1
done
```

## 5) Replay validation

```bash
python -m excavator_sim.replay --run-dir run_000
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
