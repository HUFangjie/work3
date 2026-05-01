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


def find_project_root() -> Path:
    cur = Path.cwd().resolve()
    for p in [cur, *cur.parents, Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        if (p / "observations").exists() and (p / "analysis").exists():
            return p
    return cur


def resolve_out_dir(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return ensure_dir(p)
    root = find_project_root()
    return ensure_dir((root / p).resolve())


def resolve_input_file(path_str: str) -> Path:
    p = Path(path_str)
    cands = []
    if p.is_absolute():
        cands.append(p)
    else:
        cands.append((Path.cwd() / p).resolve())
        root = find_project_root()
        cands.append((root / p).resolve())
        cands.append((Path('/') / p).resolve())

    for c in cands:
        if c.exists() and c.is_file():
            return c

    # Fallback: search by basename under project observations tree (helps when old runs wrote nested paths).
    root = find_project_root()
    matches = sorted((root / "observations").rglob(Path(path_str).name))
    file_matches = [m for m in matches if m.is_file()]
    if len(file_matches) == 1:
        print(f"[Obs1] logits file not found at requested path; fallback to discovered file: {file_matches[0]}")
        return file_matches[0]

    raise FileNotFoundError(
        "Input logits npz not found.\n"
        f"Requested: {path_str}\n"
        f"Current working directory: {Path.cwd()}\n"
        "Candidates tried:\n  - " + "\n  - ".join(str(x) for x in cands) +
        ("\nDiscovered candidates under observations/:\n  - " + "\n  - ".join(str(x) for x in file_matches[:20]) if file_matches else "")
    )
