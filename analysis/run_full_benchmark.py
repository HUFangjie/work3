# analysis/run_full_benchmark.py
"""Run benchmark experiments by calling main.py with generated overrides.

Default behavior is intentionally minimal: global training/data/model parameters
come from config/base_config.py. EXPERIMENTS only changes attack/defense and the
per-run logging name, so you can switch Tiny-ImageNet WRN-28-4/WRN-28-8 by
editing base_config.py and then running this script directly.

Usage:
    python -m analysis.run_full_benchmark

Optional CLI flags (dataset, rounds, clients, etc.) are true overrides. If they
are omitted, the values in base_config.py are preserved.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List


# ---------------------- 实验组合配置 ---------------------- #
EXPERIMENTS: List[Dict] = [
    # ---------------- Benign baselines ----------------
    {
        "name": "baseline",
        "description": "no attack, no defense",
        "attack": "none",
        "attack_enabled": False,
        "defense": "none",
        "defense_enabled": False,
    },
]


# ---------------------- 路径解析工具 ---------------------- #
def get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_main_abs_path() -> str:
    project_root = get_project_root()
    main_path = os.path.join(project_root, "main.py")
    if not os.path.isfile(main_path):
        raise FileNotFoundError(
            f"Cannot find main.py at inferred path: {main_path}\n"
            f"Check that your project structure is:\n"
            f"  {project_root}/main.py\n"
            f"  {project_root}/analysis/run_full_benchmark.py\n"
        )
    return main_path


def write_exp_override_config(base_log_dir: str, exp_prefix: str, exp: Dict) -> str:
    """Write the minimal override needed for one benchmark experiment.

    Important: this deliberately does *not* duplicate data/model/FD hyperparams.
    Those should be configured once in config/base_config.py, which is exactly
    the workflow needed for choosing wrn28_4_tiny vs wrn28_8_tiny.

    Runtime artifacts are written to a per-experiment artifact_dir under
    base_log_dir. This prevents concurrent benchmark runs on different GPUs from
    appending to the same CSV/log files when they share a code checkout.
    """
    exp_name = f"{exp_prefix}_{exp['name']}"
    artifact_dir = os.path.join(base_log_dir, exp_name)
    override: Dict = {
        "attack_config": {
            "enabled": bool(exp.get("attack_enabled", False)),
            "name": str(exp.get("attack", "none")),
        },
        "defense_config": {
            "enabled": bool(exp.get("defense_enabled", False)),
            "name": str(exp.get("defense", "none")),
        },
        "logging_config": {
            "log_dir": artifact_dir,
            "artifact_dir": artifact_dir,
            "exp_name": exp_name,
        },
    }

    os.makedirs(base_log_dir, exist_ok=True)
    os.makedirs(artifact_dir, exist_ok=True)
    path = os.path.join(base_log_dir, f"{exp_name}_override.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(override, f, indent=2, ensure_ascii=False)
    return path


# ---------------------- 组装命令行 ---------------------- #
def build_command(
    args: argparse.Namespace,
    exp: Dict,
    main_abs: str,
) -> List[str]:
    """Build the command line to run main.py for a benchmark experiment."""
    cmd: List[str] = [sys.executable, main_abs]

    # Optional overrides. If omitted, main.py uses config/base_config.py.
    if args.dataset is not None:
        cmd += ["--dataset", args.dataset]
    if args.num_rounds is not None:
        cmd += ["--num_rounds", str(args.num_rounds)]
    if args.clients_per_round is not None:
        cmd += ["--clients_per_round", str(args.clients_per_round)]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    if args.device is not None:
        cmd += ["--device", args.device]
    if args.num_clients is not None:
        cmd += ["--num_clients", str(args.num_clients)]
    if args.partition_type is not None:
        cmd += ["--partition_type", args.partition_type]
    if args.dirichlet_alpha is not None:
        cmd += ["--dirichlet_alpha", str(args.dirichlet_alpha)]

    cmd += ["--attack", exp["attack"]]
    cmd += ["--attack_enabled" if exp["attack_enabled"] else "--attack_disabled"]
    cmd += ["--defense", exp["defense"]]
    cmd += ["--defense_enabled" if exp["defense_enabled"] else "--defense_disabled"]

    override_path = write_exp_override_config(args.base_log_dir, args.exp_prefix, exp)
    cmd += ["--exp_config", override_path]
    return cmd


# ---------------------- 主入口 ---------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run FD benchmark experiments. By default, data/model/FD params are "
            "read from config/base_config.py; CLI flags are optional overrides."
        )
    )
    parser.add_argument("--dataset", type=str, default=None, help="Optional dataset override.")
    parser.add_argument("--num_rounds", type=int, default=None, help="Optional round-count override.")
    parser.add_argument("--clients_per_round", type=int, default=None, help="Optional clients-per-round override.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override.")
    parser.add_argument("--device", type=str, default=None, help='Optional device override, e.g. "cuda" or "cpu".')
    parser.add_argument("--num_clients", type=int, default=None, help="Optional total-clients override.")
    parser.add_argument("--partition_type", type=str, default=None, help="Optional partition-type override.")
    parser.add_argument("--dirichlet_alpha", type=float, default=None, help="Optional Dirichlet-alpha override.")
    parser.add_argument("--base_log_dir", type=str, default="runs/bench", help="Directory for generated overrides, command logs, and run logs.")
    parser.add_argument("--exp_prefix", type=str, default="fd_bench", help="Prefix for generated experiment names.")

    args = parser.parse_args()
    main_abs = get_main_abs_path()
    os.makedirs(args.base_log_dir, exist_ok=True)

    print("========== FD Benchmark Runner ==========")
    print(f"Project root      : {get_project_root()}")
    print(f"main.py           : {main_abs}")
    print("Config source     : config/base_config.py (unless CLI override is provided)")
    print(f"Dataset override  : {args.dataset}")
    print(f"Rounds override   : {args.num_rounds}")
    print(f"Clients override  : {args.clients_per_round}")
    print(f"Seed override     : {args.seed}")
    print(f"Base log dir      : {args.base_log_dir}")
    print(f"Exp name prefix   : {args.exp_prefix}")
    print("Artifact dirs     : <base_log_dir>/<exp_prefix>_<experiment_name>/")
    print("----------------------------------------")

    for i, exp in enumerate(EXPERIMENTS):
        print(f"[{i + 1}/{len(EXPERIMENTS)}] Running experiment: {exp['name']}")
        print(f"    Description : {exp['description']}")
        artifact_dir = os.path.join(args.base_log_dir, f"{args.exp_prefix}_{exp['name']}")
        print(f"    Artifacts   : {artifact_dir}")
        cmd = build_command(args, exp, main_abs)
        print(f"    Command     : {' '.join(cmd)}")

        cmd_log_name = f"{args.exp_prefix}_{exp['name']}_cmd.txt"
        run_log_path = os.path.join(args.base_log_dir, cmd_log_name)
        with open(run_log_path, "w", encoding="utf-8") as f:
            f.write(" ".join(cmd) + "\n")

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Experiment {exp['name']} failed with return code {e.returncode}")
            continue

    print("All experiments finished (or attempted).")


if __name__ == "__main__":
    main()
