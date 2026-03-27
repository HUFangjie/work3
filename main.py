#!/usr/bin/env python
# main.py
"""
Entry point for Federated Distillation + T3 attack + defenses.

兼容 run_full_benchmark.py 的命令行参数：
  --dataset, --num_rounds, --clients_per_round, --seed,
  --attack / --attack_enabled / --attack_disabled,
  --defense / --defense_enabled / --defense_disabled,
  --exp_name, --log_dir, --exp_config, 等。
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch

from config.base_config import get_base_config  # BASE_CONFIG + deepcopy():contentReference[oaicite:5]{index=5}

# 核心模块
from core.data_manager import DataManager
from core.client import Client
from core.server import Server
from core.federated_distillation import run_federated_distillation

from core.utils import get_device

# 模型 & 攻击 & 防御工厂（按你之前的设计）
from models.model_zoo import get_model, adapt_model_config_for_dataset
from attacks.__init__ import create_attack
from defenses.__init__ import create_defense


# ======================================================================
# 1. 命令行解析
# ======================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Federated Distillation with T3 attack and defenses"
    )

    # 额外实验配置文件（.py/.json/.yaml）
    parser.add_argument(
        "--exp_config",
        type=str,
        default=None,
        help="Optional experiment config file (.py/.json/.yaml).",
    )

    # 通用
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Device string, e.g., "cuda" or "cpu". If None, use config/default.',
    )

    # 数据 / 划分
    parser.add_argument(
        "--dataset",
        type=str,
        default="fmnist",
        help='Dataset name: "fmnist" or "cifar10".',
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=None,
        help="Total number of clients (override config).",
    )
    parser.add_argument(
        "--partition_type",
        type=str,
        default=None,
        help='Partition type: "dirichlet", "shard", "label_separation".',
    )
    parser.add_argument(
        "--dirichlet_alpha",
        type=float,
        default=None,
        help="Dirichlet alpha for non-IID partition.",
    )

    # 训练协议
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=None,
        help="Number of communication rounds (override config).",
    )
    parser.add_argument(
        "--clients_per_round",
        type=int,
        default=None,
        help="Number of clients per round (override config).",
    )

    # 攻击设置
    parser.add_argument(
        "--attack",
        type=str,
        default=None,
        help='Attack name: "none", "t3", "gaussian", "label_flip". '
             "If None, use config.attack_config.name.",
    )
    parser.add_argument(
        "--attack_enabled",
        action="store_true",
        help="Force enable attack.",
    )
    parser.add_argument(
        "--attack_disabled",
        action="store_true",
        help="Force disable attack (overrides --attack_enabled).",
    )

    # 防御设置
    parser.add_argument(
        "--defense",
        type=str,
        default=None,
        help='Defense name: "none", "cronus", "entropy_clip". '
             "If None, use config.defense_config.name.",
    )
    parser.add_argument(
        "--defense_enabled",
        action="store_true",
        help="Force enable defense.",
    )
    parser.add_argument(
        "--defense_disabled",
        action="store_true",
        help="Force disable defense (overrides --defense_enabled).",
    )

    # 日志 & 输出（兼容 run_full_benchmark）
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Experiment name (override logging_config.exp_name).",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Root log directory (override logging_config.log_dir).",
    )

    return parser.parse_args()


# ======================================================================
# 2. 配置加载与 merge
# ======================================================================
def _load_exp_config(path: str) -> Dict[str, Any]:
    """
    加载额外实验配置文件 (.py / .json / .yaml)，用于覆盖 BASE_CONFIG。
    """
    if path is None:
        return {}

    if not os.path.exists(path):
        raise FileNotFoundError(f"exp_config file not found: {path}")

    if path.endswith(".json"):
        import json

        with open(path, "r") as f:
            return json.load(f)
    elif path.endswith(".yml") or path.endswith(".yaml"):
        import yaml

        with open(path, "r") as f:
            return yaml.safe_load(f)
    elif path.endswith(".py"):
        spec = importlib.util.spec_from_file_location("exp_config_module", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)  # type: ignore
        if hasattr(module, "get_config"):
            return module.get_config()
        elif hasattr(module, "CONFIG"):
            return module.CONFIG
        else:
            raise ValueError(
                f"Python exp_config {path} must define `get_config()` or `CONFIG`."
            )
    else:
        raise ValueError(
            f"Unsupported exp_config extension for {path}; use .py/.json/.yaml."
        )


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """
    递归将 src merge 到 dst 中。
    """
    for k, v in src.items():
        if (
            k in dst
            and isinstance(dst[k], dict)
            and isinstance(v, dict)
        ):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    """
    组合：
      - base_config (get_base_config)
      - optional exp_config
      - CLI 覆盖
    """
    cfg = get_base_config()  # 深拷贝 BASE_CONFIG:contentReference[oaicite:6]{index=6}

    # 2) 额外实验配置
    if args.exp_config is not None:
        extra = _load_exp_config(args.exp_config)
        cfg = _deep_update(cfg, extra)

    # 3) CLI 覆盖
    # --- data_config ---
    data_cfg = cfg["data_config"]
    if args.dataset is not None:
        data_cfg["dataset"] = args.dataset.lower()
    if args.num_clients is not None:
        data_cfg["num_clients"] = int(args.num_clients)
    if args.partition_type is not None:
        data_cfg["partition_type"] = args.partition_type
    if args.dirichlet_alpha is not None:
        data_cfg["dirichlet_alpha"] = float(args.dirichlet_alpha)

    # --- fd_config ---
    fd_cfg = cfg["fd_config"]
    if args.num_rounds is not None:
        fd_cfg["num_rounds"] = int(args.num_rounds)
    if args.clients_per_round is not None:
        fd_cfg["clients_per_round"] = int(args.clients_per_round)

    # --- attack_config ---
    attack_cfg = cfg["attack_config"]
    if args.attack is not None:
        attack_cfg["name"] = args.attack.lower()
    # enabled 逻辑：CLI 优先
    if args.attack_disabled:
        attack_cfg["enabled"] = False
    elif args.attack_enabled:
        attack_cfg["enabled"] = True
    else:
        # 默认为非 "none" 时启用
        attack_cfg["enabled"] = attack_cfg.get("name", "none") != "none"

    # --- defense_config ---
    defense_cfg = cfg["defense_config"]
    if args.defense is not None:
        defense_cfg["name"] = args.defense.lower()
    if args.defense_disabled:
        defense_cfg["enabled"] = False
    elif args.defense_enabled:
        defense_cfg["enabled"] = True
    else:
        defense_cfg["enabled"] = defense_cfg.get("name", "none") != "none"

    # --- logging_config ---
    log_cfg = cfg["logging_config"]
    if args.log_dir is not None:
        log_cfg["log_dir"] = args.log_dir
    if args.exp_name is not None:
        log_cfg["exp_name"] = args.exp_name

    # --- seed/device ---
    cfg["seed"] = int(args.seed)
    if args.device is not None:
        cfg["device"] = args.device

    return cfg


# ======================================================================
# 3. 日志 & TensorBoard
# ======================================================================
class DummyWriter:
    def add_scalar(self, tag: str, value: float, global_step: int) -> None:
        # 安全空实现，保证调用不报错
        pass



def setup_logger_and_writer(config: Dict[str, Any]):
    log_cfg = config["logging_config"]
    log_dir = log_cfg.get("log_dir", "./logs")
    exp_name = log_cfg.get("exp_name", "debug_run")

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{exp_name}.log")

    # 实验专用 logger（保持不变，用于 main / federated_distillation 等）
    logger = logging.getLogger(exp_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # 不再向 root 冒泡，避免重复

    # 清空旧 handler，避免重复输出
    if logger.handlers:
        logger.handlers.clear()

    # ---- 统一创建一套 handler（控制台 + 文件）----
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    # 挂到实验 logger 上
    logger.addHandler(ch)
    logger.addHandler(fh)

    # ---- 同步到 root logger：保证 logging.getLogger() / getLogger(__name__)
    # ---- 打出来的日志也写入同一个文件和控制台 ----
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # 清空旧 handler，避免和其他地方的 basicConfig 冲突
    if root.handlers:
        root.handlers.clear()
    # 共享同一套 handler（注意：handler 是同一对象）
    root.addHandler(ch)
    root.addHandler(fh)

    # TensorBoard / Dummy
    if log_cfg.get("use_tensorboard", True):
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_dir = os.path.join(log_dir, exp_name)
            writer = SummaryWriter(log_dir=tb_dir)
        except Exception:
            logger.warning("TensorBoard not available; falling back to DummyWriter.")
            writer = DummyWriter()
    else:
        writer = DummyWriter()

    return logger, writer


# ======================================================================
# 4. 构建数据、客户端、服务器
# ======================================================================
def _select_malicious_clients(
    num_clients: int,
    attack_cfg: Dict[str, Any],
    rng: np.random.RandomState,
) -> np.ndarray:
    frac = float(attack_cfg.get("malicious_client_fraction", 0.0))
    fixed = attack_cfg.get("fixed_malicious_clients", None)
    if fixed is not None:
        return np.array(fixed, dtype=int)
    m = int(round(frac * num_clients))
    m = max(0, min(m, num_clients))
    if m == 0:
        return np.array([], dtype=int)
    return rng.choice(num_clients, size=m, replace=False)


def build_data_manager(config: Dict[str, Any]) -> DataManager:
    dm = DataManager(config=config)  # 会内部完成划分和 loader 构建:contentReference[oaicite:7]{index=7}
    return dm


def build_clients(
    config: Dict[str, Any],
    device: torch.device,
    data_manager: DataManager,
) -> Dict[int, Client]:
    data_cfg = config["data_config"]
    model_cfg = config["model_config"]
    attack_cfg = config["attack_config"]

    # 按数据集自动修正模型输入通道 & 类别数
    model_cfg = adapt_model_config_for_dataset(
        dataset=data_cfg["dataset"],
        model_config=model_cfg,
    )

    num_clients = data_manager.get_num_clients()
    rng = np.random.RandomState(config.get("seed", 42))
    malicious_ids = _select_malicious_clients(num_clients, attack_cfg, rng)
    malicious_flags = np.zeros(num_clients, dtype=bool)
    malicious_flags[malicious_ids] = True

    clients: Dict[int, Client] = {}

    for cid in range(num_clients):
        # 实例化本地模型
        model = get_model(
            name=model_cfg["name"],
            input_channels=model_cfg["input_channels"],
            num_classes=model_cfg["num_classes"],
            width_mult=model_cfg.get("width_mult", 1.0),
            dropout=model_cfg.get("dropout", 0.0),
        ).to(device)

        private_loader = data_manager.get_client_private_loader(cid)

        # 攻击模块（只为恶意客户端 & attack.enabled 时创建）
        if attack_cfg.get("enabled", False) and malicious_flags[cid]:
            attack = create_attack(
                attack_config=attack_cfg,
                client_id=cid,
                is_malicious=True,
                model=model,
                dataset_name=data_cfg["dataset"],
            )
        else:
            attack = None

        client = Client(
            client_id=cid,
            model=model,
            private_loader=private_loader,
            device=device,
            fd_config=config["fd_config"],
            attack=attack,
        )
        clients[cid] = client

    return clients


def build_server(
    config: Dict[str, Any],
    device: torch.device,
) -> Server:
    defense_cfg = config["defense_config"]
    defense = create_defense(defense_cfg, device=device)

    # ---- NEW: server-side real global student model ----
    data_cfg = config["data_config"]
    model_cfg = adapt_model_config_for_dataset(
        dataset=data_cfg["dataset"],
        model_config=config["model_config"],
    )
    student_model = get_model(
        name=model_cfg["name"],
        input_channels=model_cfg["input_channels"],
        num_classes=model_cfg["num_classes"],
        width_mult=model_cfg.get("width_mult", 1.0),
        dropout=model_cfg.get("dropout", 0.0),
    ).to(device)

    server = Server(
        device=device,
        defense=defense,
        student_model=student_model,
        fd_config=config["fd_config"],
    )
    return server

# ======================================================================
# 5. 主入口
# ======================================================================
def main():
    args = parse_args()
    config = build_config(args)

    # 设定随机种子
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 设备
    device = get_device(config)

    logger, writer = setup_logger_and_writer(config)

    logger.info("===== Federated Distillation: Config Summary =====")
    logger.info(f"Dataset            : {config['data_config']['dataset']}")
    logger.info(f"Num clients        : {config['data_config']['num_clients']}")
    logger.info(f"Partition type     : {config['data_config'].get('partition_type', 'dirichlet')}")
    logger.info(f"Dirichlet alpha    : {config['data_config'].get('dirichlet_alpha', 'N/A')}")
    logger.info(f"Num rounds         : {config['fd_config']['num_rounds']}")
    logger.info(f"Clients per round  : {config['fd_config']['clients_per_round']}")
    logger.info(f"Attack enabled     : {config['attack_config']['enabled']}")
    logger.info(f"Attack name        : {config['attack_config']['name']}")
    logger.info(f"Defense enabled    : {config['defense_config']['enabled']}")
    logger.info(f"Defense name       : {config['defense_config']['name']}")
    logger.info(f"Device             : {device}")
    logger.info("=================================================")

    # 数据管理器
    data_manager = build_data_manager(config)
    logger.info(data_manager.summary())

    # 客户端
    clients = build_clients(config, device, data_manager)

    # 服务器
    server = build_server(config, device)

    # 公共 / 验证 / 测试 loader
    public_loader = data_manager.get_public_loader()
    val_loader = data_manager.get_val_loader()
    test_loader = data_manager.get_test_loader()

    # 主训练循环（内部已支持 attack/defense）
    run_federated_distillation(
        config=config,
        server=server,
        clients=clients,
        public_loader=public_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        logger=logger,
        writer=writer,
    )

    logger.info("Training finished.")


if __name__ == "__main__":
    main()
