"""Export real client logits from current federated pipeline for observations.

Outputs:
- benign_logits.npy: shape (K, N_pub, C)
- obs1_logits.npz: keys like alpha_0.5_seed_1, values shape (K, N_pub, C)
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

# Some environments set invalid OMP_NUM_THREADS values (e.g., empty / non-integer),
# which causes libgomp to abort at runtime. Force a safe default when invalid.
_omp = os.environ.get("OMP_NUM_THREADS", "").strip()
if (not _omp.isdigit()) or int(_omp) <= 0:
    os.environ["OMP_NUM_THREADS"] = "1"

# Ensure project root is importable even when launched outside repository root.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch

from config.base_config import get_base_config
from core.federated_distillation import run_federated_distillation
from main import build_clients, build_data_manager, build_server



def _resolve_output_path(path_str: str, run_tag: str, auto_suffix: bool) -> Path:
    p = Path(path_str)
    if run_tag:
        p = p.with_name(f"{p.stem}_{run_tag}{p.suffix}")
    if auto_suffix and p.exists():
        idx = 1
        while True:
            cand = p.with_name(f"{p.stem}_v{idx}{p.suffix}")
            if not cand.exists():
                p = cand
                break
            idx += 1
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

class _NoopWriter:
    def add_scalar(self, *args, **kwargs):
        return None


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_config(
    dataset: str,
    num_clients: int,
    alpha: float,
    seed: int,
    device: str,
    num_rounds: int,
    local_epochs: int,
    clients_per_round: int,
    data_root: str,
    allow_download: bool,
) -> Dict:
    cfg = get_base_config()
    cfg["seed"] = int(seed)
    cfg["device"] = device

    cfg["data_config"]["dataset"] = dataset
    cfg["data_config"]["num_clients"] = int(num_clients)
    cfg["data_config"]["partition_type"] = "dirichlet"
    cfg["data_config"]["data_root"] = data_root
    cfg["data_config"]["download"] = bool(allow_download)
    cfg["data_config"]["dirichlet_alpha"] = float(alpha)

    cfg["fd_config"]["num_rounds"] = int(num_rounds)
    cfg["fd_config"]["local_epochs"] = int(local_epochs)
    cfg["fd_config"]["clients_per_round"] = int(clients_per_round)

    # Ensure benign run for real benign logits export.
    cfg["attack_config"]["enabled"] = False
    cfg["defense_config"]["enabled"] = False
    cfg["defense_config"]["name"] = "none"

    # keep exporter quiet and lightweight
    cfg["logging_config"]["use_tensorboard"] = False
    cfg["evaluation_config"]["eval_every"] = max(1_000_000, int(num_rounds) + 1)
    return cfg


def _collect_client_logits(clients: Dict[int, object], public_loader, device: torch.device) -> np.ndarray:
    ids = sorted(clients.keys())
    per_client_chunks: Dict[int, List[np.ndarray]] = {cid: [] for cid in ids}

    for x_pub, y_pub in public_loader:
        x_pub = x_pub.to(device, non_blocking=True)
        y_pub = y_pub.to(device, non_blocking=True) if torch.is_tensor(y_pub) else y_pub
        for cid in ids:
            logits = clients[cid].compute_public_logits(x_public=x_pub, y_public=y_pub, round_idx=None)
            per_client_chunks[cid].append(logits.detach().float().cpu().numpy())

    stacked = []
    for cid in ids:
        arr = np.concatenate(per_client_chunks[cid], axis=0)
        stacked.append(arr)
    return np.stack(stacked, axis=0)


def _train_and_export_one(cfg: Dict, out_benign_npy: Path | None) -> np.ndarray:
    seed = int(cfg["seed"])
    _set_seed(seed)

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    logger = logging.getLogger(f"export_real_logits_seed_{seed}")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.addHandler(logging.StreamHandler())
    writer = _NoopWriter()

    dm = build_data_manager(cfg)
    clients = build_clients(cfg, device, dm)
    server = build_server(cfg, device)

    run_federated_distillation(
        config=cfg,
        server=server,
        clients=clients,
        public_loader=dm.get_public_loader(),
        val_loader=dm.get_val_loader(),
        test_loader=dm.get_test_loader(),
        logger=logger,
        writer=writer,
    )

    logits = _collect_client_logits(clients, dm.get_public_loader(), device=device)
    if out_benign_npy is not None:
        out_benign_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_benign_npy, logits)
    return logits


def main() -> None:
    parser = argparse.ArgumentParser(description="Export real logits for ReG-Trust observations.")
    parser.add_argument("--dataset", default="cifar10", help="Dataset name (e.g., cifar10/fmnist)")
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--clients-per-round", type=int, default=20)
    parser.add_argument("--num-rounds", type=int, default=20)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-root", default="./data", help="Dataset root passed to DataManager (e.g., analysis/data)")
    parser.add_argument("--allow-download", action="store_true", help="Allow auto-download when dataset files are missing")

    parser.add_argument("--benign-alpha", type=float, default=0.5, help="Alpha used to export benign_logits.npy")
    parser.add_argument("--benign-seed", type=int, default=42)
    parser.add_argument("--out-benign-npy", default="observations/real_inputs/benign_logits.npy")

    parser.add_argument("--obs1-alphas", nargs="+", type=float, default=[0.1, 0.3, 0.5, 1.0])
    parser.add_argument("--obs1-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--obs1-seed-offset", type=int, default=1000)
    parser.add_argument("--out-obs1-npz", default="observations/real_inputs/obs1_logits.npz")
    parser.add_argument("--run-tag", default="", help="Optional tag appended to output filenames")
    parser.add_argument("--auto-suffix", action="store_true", help="If output exists, append _vN to avoid overwrite")
    args = parser.parse_args()

    out_benign = _resolve_output_path(args.out_benign_npy, run_tag=args.run_tag, auto_suffix=args.auto_suffix)
    out_obs1 = _resolve_output_path(args.out_obs1_npz, run_tag=args.run_tag, auto_suffix=args.auto_suffix)

    print(f"[export_real_logits] benign output: {out_benign}")
    print(f"[export_real_logits] obs1 output  : {out_obs1}")

    # 1) benign_logits.npy
    benign_cfg = _build_config(
        dataset=args.dataset,
        num_clients=args.num_clients,
        alpha=args.benign_alpha,
        seed=args.benign_seed,
        device=args.device,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        clients_per_round=args.clients_per_round,
        data_root=args.data_root,
        allow_download=args.allow_download,
    )
    _train_and_export_one(benign_cfg, out_benign_npy=out_benign)

    # 2) obs1_logits.npz
    obs1_payload = {}
    for alpha in args.obs1_alphas:
        for s in args.obs1_seeds:
            run_seed = int(args.obs1_seed_offset + s)
            cfg = _build_config(
                dataset=args.dataset,
                num_clients=args.num_clients,
                alpha=float(alpha),
                seed=run_seed,
                device=args.device,
                num_rounds=args.num_rounds,
                local_epochs=args.local_epochs,
                clients_per_round=args.clients_per_round,
                data_root=args.data_root,
                allow_download=args.allow_download,
            )
            logits = _train_and_export_one(cfg, out_benign_npy=None)
            obs1_payload[f"alpha_{alpha}_seed_{s}"] = logits

    np.savez_compressed(out_obs1, **obs1_payload)


if __name__ == "__main__":
    main()
