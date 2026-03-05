from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path

from excavator_sim.common import get_paths


def _python_cmd(module: str):
    return [sys.executable, "-m", module]


def parse_args():
    paths = get_paths()
    parser = argparse.ArgumentParser(description="Launch sim + teleop + recorder workflow")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--record-base", default=str(paths.raw_data))
    parser.add_argument("--mode", default="position", choices=["position", "velocity"])
    return parser.parse_args()


def main():
    args = parse_args()
    env = os.environ.copy()

    sim_cmd = _python_cmd("excavator_sim.run_sim") + (["--headless"] if args.headless else [])
    rec_cmd = _python_cmd("excavator_sim.ros.record_topics") + ["--base-dir", str(args.record_base)]
    teleop_cmd = _python_cmd("excavator_sim.teleop") + ["--mode", args.mode]

    procs = [
        subprocess.Popen(sim_cmd, env=env),
        subprocess.Popen(rec_cmd, env=env),
        subprocess.Popen(teleop_cmd, env=env),
    ]

    try:
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            if p.poll() is None:
                p.send_signal(signal.SIGINT)
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    main()
