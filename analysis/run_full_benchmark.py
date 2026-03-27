# analysis/run_full_benchmark.py
"""
Run a small grid of experiments (benign vs T3, defense none vs cronus)
by calling main.py with different configs.

Usage (from ANY directory under the project, 推荐在项目根目录):
    python -m analysis.run_full_benchmark \
        --dataset fmnist \
        --num_rounds 20 \
        --clients_per_round 10 \
        --base_log_dir runs/bench_fmnist

脚本会自动寻找 main.py 的绝对路径，不再依赖当前工作目录。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Dict, List


# ---------------------- 实验组合配置 ---------------------- #
EXPERIMENTS: List[Dict] = [
    # ---------------- Benign baselines ----------------
#     {
#         "name": "benign_none",
#         "description": "No attack, no defense",
#         "attack": "none",
#         "attack_enabled": False,
#         "defense": "none",
#         "defense_enabled": False,
#     },

#     # ---------------- Attacks (no defense) ----------------
#     {
#         "name": "t3_none",
#         "description": "T3 attack, no defense",
#         "attack": "t3",
#         "attack_enabled": True,
#         "defense": "none",
#         "defense_enabled": False,
#     },
#     {
#         "name": "topk_none",
#         "description": "TopK attack, no defense",
#         "attack": "topk",
#         "attack_enabled": True,
#         "defense": "none",
#         "defense_enabled": False,
#     },
    {
        "name": "fed_ace",
        "description": "fed_ace, trimean defense",
        "attack": "fed_ace",
        "attack_enabled": True,
        "defense": "trimean",
        "defense_enabled": True,
    },
#     {
#         "name": "manipulating_kd_none",
#         "description": "ManipulatingKD attack, no defense",
#         "attack": "manipulating_kd",
#         "attack_enabled": True,
#         "defense": "none",
#         "defense_enabled": False,
#     },
#     {
#         "name": "fed_ace_none",
#         "description": "Fed-ACE attack, no defense",
#         "attack": "fed_ace",
#         "attack_enabled": True,
#         "defense": "none",
#         "defense_enabled": False,
#     },
#     {
#         "name": "fed_oca_none",
#         "description": "Fed-OCA attack, no defense",
#         "attack": "fed_oca",
#         "attack_enabled": True,
#         "defense": "none",
#         "defense_enabled": False,
#     },

#     # ---------------- Defenses under attack ----------------
#     # T3 + defenses
#     {
#         "name": "t3_mkrum",
#         "description": "T3 attack + MKrum defense",
#         "attack": "t3",
#         "attack_enabled": True,
#         "defense": "mkrum",
#         "defense_enabled": True,
#     },
#     {
#         "name": "t3_trimean",
#         "description": "T3 attack + TriMean defense",
#         "attack": "t3",
#         "attack_enabled": True,
#         "defense": "trimean",
#         "defense_enabled": True,
#     },
#     {
#         "name": "t3_fedmdr",
#         "description": "T3 attack + FedMDR defense",
#         "attack": "t3",
#         "attack_enabled": True,
#         "defense": "fedmdr",
#         "defense_enabled": True,
#     },
#     {
#         "name": "t3_fedtgd",
#         "description": "T3 attack + FedTGD defense",
#         "attack": "t3",
#         "attack_enabled": True,
#         "defense": "fedtgd",
#         "defense_enabled": True,
#     },

#     # Fed-OCA + defenses (example)
#     {
#         "name": "fed_oca_mkrum",
#         "description": "Fed-OCA attack + MKrum defense",
#         "attack": "fed_oca",
#         "attack_enabled": True,
#         "defense": "mkrum",
#         "defense_enabled": True,
#     },
]


# ---------------------- 路径解析工具 ---------------------- #
def get_main_abs_path() -> str:
    """
    通过当前文件位置，自动推断 main.py 的绝对路径。

    假设目录结构类似：
        project_root/
            main.py
            analysis/
                run_full_benchmark.py
            ...

    则 __file__ 在 project_root/analysis/ 下，
    project_root = dirname(dirname(__file__))
    """
    this_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(this_file))
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
    """
    Write a minimal JSON override config for this experiment, so we can set
    per-attack / per-defense hyperparameters without touching the global base_config.py.

    Returns the path to the generated JSON file.
    """
    override: Dict = {
        "attack_config": {
            "enabled": bool(exp.get("attack_enabled", False)),
            "name": str(exp.get("attack", "none")),
            # ---- Attack hyperparam templates (edit as needed) ----
            "naive_sharpening": {"temperature": 0.5, "max_abs_logit": None},
            "manipulating_kd": {"gamma": 0.5, "min_margin": 1e-3, "max_margin": 50.0, "l2_budget": None, "seed": 1234},
            "fed_ace": {"tau": 0.7, "gamma": 0.5, "min_margin": 1e-3, "max_margin": 50.0, "l2_budget": None},
            "fed_oca": {"gamma": 0.8, "min_margin": 1e-3, "max_margin": 50.0, "l2_budget": None},
            # keep existing attacks' keys if your main.py merges them
            "t3": {
                "rho": 0.4,
                "lambda_epistemic": 1.0,
                "lambda_stealth": 1.0,
                "lambda_align": 1.0,
                "epsilon": 2.0,
                "pgd_steps": 40,
                "pgd_step_size": 0.2,
                "history_window": 1,
                "tta_type": "strong",
                "debug": False,
            },
            "topk": {"k": 3, "delta": -10.0, "normalize": True, "norm_low": -10.0, "norm_high": 10.0},
            "gaussian": {"sigma": 0.1},
            "label_flip": {"flip_probability": 0.5},
            "impersonation": {},
        },
        "defense_config": {
            "enabled": bool(exp.get("defense_enabled", False)),
            "name": str(exp.get("defense", "none")),
            # ---- Defense hyperparam templates (edit as needed) ----
            "mkrum": {"f": 1, "m": 1, "use_squared": True},
            "trimean": {"q": 0.25},
            "fedmdr": {"trim_ratio": 0.2, "softmax_temp": 1.0, "max_iter": 50, "eps": 1e-6, "min_clients_kept": 2},
            "fedtgd": {"topk": 5, "dbscan_eps": 0.5, "dbscan_min_samples": 2, "cosine_keep_ratio": 1.0, "fallback": "mean"},
            "cronus": {"temperature": 1.0, "gamma": 2.0, "trimming_fraction": 0.2, "min_clients_kept": 2},
            "entropy_clip": {"max_entropy": 2.5},
            "none": {},
        },
    }

    os.makedirs(base_log_dir, exist_ok=True)
    filename = f"{exp_prefix}_{exp['name']}_override.json"
    path = os.path.join(base_log_dir, filename)
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(override, f, indent=2, ensure_ascii=False)
    return path


# ---------------------- 组装命令行 ---------------------- #
def build_command(
    args: argparse.Namespace,
    exp: Dict,
    main_abs: str,
) -> List[str]:
    """
    Build the command line to run main.py for a given experiment config.

    严格只使用 main.py usage 中列出来的参数：
      --exp_config, --seed, --device, --dataset, --num_clients,
      --partition_type, --dirichlet_alpha, --num_rounds,
      --clients_per_round, --attack, --attack_enabled/--attack_disabled,
      --defense, --defense_enabled/--defense_disabled
    """
    cmd: List[str] = [sys.executable, main_abs]

    # 必要参数：dataset & 训练轮次 & 每轮客户端数
    cmd += ["--dataset", args.dataset]
    cmd += ["--num_rounds", str(args.num_rounds)]
    cmd += ["--clients_per_round", str(args.clients_per_round)]

    # 选填：seed
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]

    # 选填：device（如果 main.py 支持 "cuda" / "cpu"）
    if args.device is not None:
        cmd += ["--device", args.device]

    # 选填：num_clients
    if args.num_clients is not None:
        cmd += ["--num_clients", str(args.num_clients)]

    # 选填：partition_type
    if args.partition_type is not None:
        cmd += ["--partition_type", args.partition_type]

    # 选填：dirichlet_alpha
    if args.dirichlet_alpha is not None:
        cmd += ["--dirichlet_alpha", str(args.dirichlet_alpha)]

    # 攻击配置
    cmd += ["--attack", exp["attack"]]
    if exp["attack_enabled"]:
        cmd += ["--attack_enabled"]
    else:
        cmd += ["--attack_disabled"]

    # 防御配置
    cmd += ["--defense", exp["defense"]]
    if exp["defense_enabled"]:
        cmd += ["--defense_enabled"]
    else:
        cmd += ["--defense_disabled"]

    # 为每个实验自动生成一个 override config（包含新 attack/defense 的默认超参模板）
    override_path = write_exp_override_config(args.base_log_dir, args.exp_prefix, exp)
    cmd += ["--exp_config", override_path]

    return cmd


# ---------------------- 主入口 ---------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Run a small benchmark grid of FD experiments "
        "(benign vs T3, defense none vs cronus)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        help="Dataset name: fmnist or cifar10 or tiny_imagenet or pathmnist.",
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=100,
        help="Number of communication rounds.",
    )
    parser.add_argument(
        "--clients_per_round",
        type=int,
        default=10,
        help="Number of clients sampled per round.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (passed to main.py).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Device for main.py, e.g., "cuda" or "cpu". If None, let main.py decide.',
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=None,
        help="Total number of clients (if you want to override default in config).",
    )
    parser.add_argument(
        "--partition_type",
        type=str,
        default=None,
        help='Data partition type, e.g., "dirichlet" (if supported by main.py).',
    )
    parser.add_argument(
        "--dirichlet_alpha",
        type=float,
        default=None,
        help="Dirichlet alpha for non-IID partition (if supported).",
    )
    parser.add_argument(
        "--base_log_dir",
        type=str,
        default="runs/bench",
        help="Base directory to store command logs (purely for bookkeeping).",
    )
    parser.add_argument(
        "--exp_prefix",
        type=str,
        default="fd_bench",
        help="Prefix for experiment names (used only in cmd log file names).",
    )
    # 如果你打算用不同 exp_config，可以在这里添加：
    # parser.add_argument("--exp_config", type=str, default=None)

    args = parser.parse_args()

    # 自动解析 main.py 绝对路径
    main_abs = get_main_abs_path()

    os.makedirs(args.base_log_dir, exist_ok=True)

    print("========== FD Benchmark Runner ==========")
    print(f"Project root      : {os.path.dirname(main_abs)}")
    print(f"main.py           : {main_abs}")
    print(f"Dataset           : {args.dataset}")
    print(f"Num rounds        : {args.num_rounds}")
    print(f"Clients/round     : {args.clients_per_round}")
    print(f"Seed              : {args.seed}")
    print(f"Base log dir      : {args.base_log_dir}  (only for command logs)")
    print(f"Exp name prefix   : {args.exp_prefix}")
    print("----------------------------------------")

    for i, exp in enumerate(EXPERIMENTS):
        print(f"[{i+1}/{len(EXPERIMENTS)}] Running experiment: {exp['name']}")
        print(f"    Description : {exp['description']}")
        cmd = build_command(args, exp, main_abs)
        print(f"    Command     : {' '.join(cmd)}")

        # 把命令记录到文件里，方便之后复现
        cmd_log_name = f"{args.exp_prefix}_{exp['name']}_ds-{args.dataset}_cmd.txt"
        run_log_path = os.path.join(args.base_log_dir, cmd_log_name)
        with open(run_log_path, "w") as f:
            f.write(" ".join(cmd) + "\n")

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Experiment {exp['name']} failed with return code {e.returncode}")
            continue

    print("All experiments finished (or attempted).")


if __name__ == "__main__":
    main()
