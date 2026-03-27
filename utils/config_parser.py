# utils/config_parser.py
"""
Configuration parser utilities.

This module:
  1) Loads the base config from config/base_config.py
  2) Optionally loads an experiment-specific YAML config file
  3) Parses command-line arguments
  4) Merges everything into a single final configuration dictionary

Usage (in main.py):

    from utils.config_parser import load_config

    config = load_config()
"""

import argparse
import os
from typing import Any, Dict, Optional

from config.base_config import get_base_config

try:
    import yaml
except ImportError:
    yaml = None  # YAML support is optional; warn if user tries to use it.


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively update dict `base` with values from `override`.

    For any key:
      - If both base[key] and override[key] are dicts: recurse
      - Else: base[key] is replaced by override[key]
    """
    for k, v in override.items():
        if (
            k in base
            and isinstance(base[k], dict)
            and isinstance(v, dict)
        ):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _load_yaml_config(path: str) -> Dict[str, Any]:
    """
    Load a YAML config from the given path.
    """
    if yaml is None:
        raise ImportError(
            "PyYAML is not installed, but a YAML exp_config was provided. "
            "Please install with `pip install pyyaml`."
        )
    if not os.path.isfile(path):
        raise FileNotFoundError(f"YAML config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"YAML config must be a mapping, got: {type(cfg)}")
    return cfg


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build an ArgumentParser that allows overriding key config options.

    We don't expose every single nested config here, only the most
    commonly toggled ones. For more complex experiments, use YAML.
    """
    parser = argparse.ArgumentParser(
        description="Fed-T3-FD: Federated Distillation with T3 Attack"
    )
    # General
    parser.add_argument("--exp_config", type=str, default=None,
                        help="Path to YAML experiment config file.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (overrides base_config).")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use: 'cuda' or 'cpu'.")

    # Data
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name: 'femnist' or 'cifar10'.")
    parser.add_argument("--num_clients", type=int, default=None,
                        help="Number of clients.")
    parser.add_argument("--partition_type", type=str, default=None,
                        help="Partition type: 'dirichlet', 'shard', 'label_separation'.")
    parser.add_argument("--dirichlet_alpha", type=float, default=None,
                        help="Dirichlet alpha for Non-IID partition.")

    # Federated distillation
    parser.add_argument("--num_rounds", type=int, default=None,
                        help="Number of federated distillation rounds.")
    parser.add_argument("--clients_per_round", type=int, default=None,
                        help="Number of clients sampled per round.")

    # Attack
    parser.add_argument("--attack", type=str, default=None,
                        help="Attack name: 'none', 't3', 'gaussian', 'label_flip'.")
    parser.add_argument("--attack_enabled", action="store_true",
                        help="Enable attack (overrides base_config).")
    parser.add_argument("--attack_disabled", action="store_true",
                        help="Disable attack (overrides base_config).")

    # Defense
    parser.add_argument("--defense", type=str, default=None,
                        help="Defense name: 'none', 'cronus', 'entropy_clip'.")
    parser.add_argument("--defense_enabled", action="store_true",
                        help="Enable defense (overrides base_config).")
    parser.add_argument("--defense_disabled", action="store_true",
                        help="Disable defense (overrides base_config).")

    return parser


def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Override fields in the config dict using parsed CLI arguments.
    """
    # Seed & device
    if args.seed is not None:
        config["seed"] = args.seed
    if args.device is not None:
        config["device"] = args.device

    # Data
    if args.dataset is not None:
        config["data_config"]["dataset"] = args.dataset
        # Optionally adjust num_classes & input_channels in model_config later in main
    if args.num_clients is not None:
        config["data_config"]["num_clients"] = args.num_clients
    if args.partition_type is not None:
        config["data_config"]["partition_type"] = args.partition_type
    if args.dirichlet_alpha is not None:
        config["data_config"]["dirichlet_alpha"] = args.dirichlet_alpha

    # FD
    if args.num_rounds is not None:
        config["fd_config"]["num_rounds"] = args.num_rounds
    if args.clients_per_round is not None:
        config["fd_config"]["clients_per_round"] = args.clients_per_round

    # Attack
    if args.attack is not None:
        config["attack_config"]["name"] = args.attack
        config["attack_config"]["enabled"] = (args.attack != "none")

    if args.attack_enabled:
        config["attack_config"]["enabled"] = True
    if args.attack_disabled:
        config["attack_config"]["enabled"] = False

    # Defense
    if args.defense is not None:
        config["defense_config"]["name"] = args.defense
        config["defense_config"]["enabled"] = (args.defense != "none")

    if args.defense_enabled:
        config["defense_config"]["enabled"] = True
    if args.defense_disabled:
        config["defense_config"]["enabled"] = False

    return config


def load_config(cli_args: Optional[argparse.Namespace] = None) -> Dict[str, Any]:
    """
    High-level entry point used by main.py.

    1) Get base config
    2) Optionally merge YAML exp_config
    3) Parse CLI args (if not provided)
    4) Apply CLI overrides
    """
    config = get_base_config()

    parser = build_arg_parser()
    if cli_args is None:
        args = parser.parse_args()
    else:
        args = cli_args

    # Merge YAML experiment config if provided
    if args.exp_config is not None:
        yaml_cfg = _load_yaml_config(args.exp_config)
        config = _deep_update(config, yaml_cfg)

    # Apply CLI overrides
    config = apply_cli_overrides(config, args)

    return config
