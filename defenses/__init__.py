# defenses/__init__.py
"""
Defense factory: create defense objects based on config.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from defenses.base_defense import BaseDefense
from defenses.defense_none import NoDefense
from defenses.defense_entropy_clip import EntropyClipDefense
from defenses.cronus_defense import CronusDefense
from defenses.defense_mkrum import MKrumDefense
from defenses.defense_trimean import TriMeanDefense
from defenses.defense_fedmdr import FedMDRDefense
from defenses.defense_fedtgd import FedTGDDefense
from defenses.defense_confidence_aware import ConfidenceAwareDefense


def create_defense(
    defense_config: Dict[str, Any],
    device: torch.device,
) -> BaseDefense:
    """
    Create a defense object based on defense_config.

    Expected defense_config structure (example):

        "defense_config": {
            "enabled": true,
            "name": "cronus",   # or "none", "entropy_clip"
            "cronus": {
                "temperature": 1.0,
                "gamma": 2.0,
                "trimming_fraction": 0.2,
                "min_clients_kept": 2
            },
            "entropy_clip": {
                "max_entropy": 2.5
            }
        }

    Args:
        defense_config: config["defense_config"] dict.
        device: torch.device.

    Returns:
        BaseDefense subclass instance.
    """
    enabled = defense_config.get("enabled", False)
    name = defense_config.get("name", "none").lower()

    if (not enabled) or name == "none":
        return NoDefense(device=device)

    if name == "entropy_clip":
        ec_cfg = defense_config.get("entropy_clip", {})
        max_entropy = ec_cfg.get("max_entropy", 2.5)
        return EntropyClipDefense(device=device, max_entropy=max_entropy)

    if name == "cronus":
        c_cfg = defense_config.get("cronus", {})
        temperature = c_cfg.get("temperature", 1.0)
        gamma = c_cfg.get("gamma", 2.0)
        trimming_fraction = c_cfg.get("trimming_fraction", 0.0)
        min_clients_kept = c_cfg.get("min_clients_kept", 2)
        return CronusDefense(
            device=device,
            temperature=temperature,
            gamma=gamma,
            trimming_fraction=trimming_fraction,
            min_clients_kept=min_clients_kept,
        )

    if name == "mkrum":
        cfg = defense_config.get("mkrum", {}) or {}
        return MKrumDefense(
            device=device,
            byz_frac=float(cfg.get("byz_frac", 0.2)),
            f=cfg.get("f", None),
            m=cfg.get("m", None),
        )

    if name == "trimean":
        return TriMeanDefense(device=device)

    if name == "fedmdr":
        cfg = defense_config.get("fedmdr", {}) or {}
        return FedMDRDefense(
            device=device,
            rho=float(cfg.get("rho", 10.0)),
            max_iter=int(cfg.get("max_iter", 50)),
            eps=float(cfg.get("eps", 1e-6)),
            trim_on_weights=bool(cfg.get("trim_on_weights", True)),
        )

    if name == "fedtgd":
        cfg = defense_config.get("fedtgd", {}) or {}
        return FedTGDDefense(
            device=device,
            k=int(cfg.get("k", 3)),
            eps=float(cfg.get("eps", 5.0)),
            min_samples=int(cfg.get("min_samples", 1)),
            normalize_logits=bool(cfg.get("normalize_logits", True)),
        )

    if name == "confidence_aware":
        cfg = defense_config.get("confidence_aware", {}) or {}
        return ConfidenceAwareDefense(
            device=device,
            tau_conf=float(cfg.get("tau_conf", 0.9)),
            hist_window=int(cfg.get("hist_window", 5)),
            beta=float(cfg.get("beta", 2.0)),
            eps=float(cfg.get("eps", 1e-12)),
            lambdas=cfg.get("lambdas", None),
        )

    # 未知防御类型，fallback 到最简单的平均
    return NoDefense(device=device)
