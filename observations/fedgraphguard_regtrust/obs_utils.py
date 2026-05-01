from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def make_run_id(dataset: str, obs_name: str = "obs1") -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    short = hashlib.md5(f"{dataset}_{obs_name}_{ts}".encode("utf-8")).hexdigest()[:6]
    return f"{dataset}_{obs_name}_{ts}_{short}"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_obs1_key(key: str) -> Dict[str, str]:
    # dataset_cifar10_mode_homogeneous_beta_0.1_seed_0
    parts = key.split("_")
    if len(parts) < 8 or parts[0] != "dataset" or parts[2] != "mode" or parts[4] != "beta" or parts[6] != "seed":
        raise ValueError(
            f"Invalid key format: {key}. Expected dataset_<name>_mode_<mode>_beta_<beta>_seed_<seed>."
        )
    return {
        "dataset": parts[1],
        "model_mode": parts[3],
        "beta": parts[5],
        "seed": parts[7],
    }


def save_json(path: Path, obj: Dict) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
