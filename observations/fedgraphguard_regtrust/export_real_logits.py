from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

_omp = os.environ.get("OMP_NUM_THREADS", "").strip()
if (not _omp.isdigit()) or int(_omp) <= 0:
    os.environ["OMP_NUM_THREADS"] = "1"

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent))
    from obs_datasets import get_dataset
    from obs_model_factory import assign_architectures, build_model
    from obs_utils import make_run_id, save_json, resolve_out_dir
else:
    from .obs_datasets import get_dataset
    from .obs_model_factory import assign_architectures, build_model
    from .obs_utils import make_run_id, save_json, resolve_out_dir

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

def dirichlet_partition(labels: np.ndarray, num_clients: int, beta: float, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    client_indices = [[] for _ in range(num_clients)]
    for c in classes:
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        props = rng.dirichlet(np.ones(num_clients) * beta)
        cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
        splits = np.split(idx, cuts)
        for i, sp in enumerate(splits):
            client_indices[i].extend(sp.tolist())
    return [np.array(x, dtype=int) for x in client_indices]


def _collect_labels(dataset) -> np.ndarray:
    if hasattr(dataset, "targets"):
        return np.array(dataset.targets)
    return np.array([y for _, y in dataset.samples], dtype=int)


def _train_one_client(model, loader, device, epochs=1):
    model.to(device)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()


def _public_logits(model, loader, device):
    model.eval()
    outs = []
    with torch.no_grad():
        for x, _ in loader:
            outs.append(model(x.to(device)).cpu().numpy())
    return np.concatenate(outs, axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cifar10", choices=["mnist", "cifar10", "tinyimagenet"])
    p.add_argument("--data-root", default="analysis/data")
    p.add_argument("--out-dir", default="observations/real_inputs/obs1")
    p.add_argument("--num-clients", type=int, default=20)
    p.add_argument("--num-public", type=int, default=1000)
    p.add_argument("--betas", nargs="+", type=float, default=[0.1, 0.3, 0.5, 1.0])
    p.add_argument("--model-modes", nargs="+", default=["homogeneous", "heterogeneous"], choices=["homogeneous", "heterogeneous"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--check-data-only", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"[Obs1] dataset = {args.dataset}")
    print(f"[Obs1] requested data_root = {args.data_root}")
    print(f"[Obs1] output dir = {args.out_dir}")
    print("[Obs1] download = disabled")

    ds, num_classes, in_ch, _, resolved_root = get_dataset(args.dataset, args.data_root)
    print(f"[Obs1] resolved data_root = {resolved_root}")
    print(f"[Data] num_samples = {len(ds)}")
    print(f"[Data] num_classes = {num_classes}")
    print(f"[Data] input_channels = {in_ch}")

    if args.check_data_only:
        print("[Obs1] check-data-only passed.")
        return
    labels = _collect_labels(ds)

    out_dir = resolve_out_dir(args.out_dir)
    print(f"[Obs1] resolved output dir = {out_dir}")
    run_id = make_run_id(args.dataset, obs_name="obs1")
    npz_path = out_dir / f"{run_id}_logits.npz"
    meta_path = out_dir / f"{run_id}_metadata.json"

    payload: Dict[str, np.ndarray] = {}

    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        all_idx = np.arange(len(ds))
        rng.shuffle(all_idx)
        public_idx = all_idx[: args.num_public]
        train_idx = all_idx[args.num_public :]

        public_loader = DataLoader(Subset(ds, public_idx), batch_size=64, shuffle=False, num_workers=2)
        train_labels = labels[train_idx]

        for beta in args.betas:
            part_local = dirichlet_partition(train_labels, args.num_clients, beta, seed)
            global_parts = [train_idx[idx] for idx in part_local]

            for mode in args.model_modes:
                archs = assign_architectures(args.num_clients, mode=mode, seed=seed)
                logits_all = []
                for cid in range(args.num_clients):
                    model = build_model(archs[cid], in_ch=in_ch, num_classes=num_classes)
                    c_loader = DataLoader(Subset(ds, global_parts[cid]), batch_size=64, shuffle=True, num_workers=2)
                    _train_one_client(model, c_loader, device, epochs=args.local_epochs)
                    logits_all.append(_public_logits(model, public_loader, device))

                key = f"dataset_{args.dataset}_mode_{mode}_beta_{beta}_seed_{seed}"
                payload[key] = np.stack(logits_all, axis=0)

    np.savez_compressed(npz_path, **payload)
    save_json(
        meta_path,
        {
            "dataset": args.dataset,
            "data_root": args.data_root,
            "num_clients": args.num_clients,
            "num_public": args.num_public,
            "betas": args.betas,
            "seeds": args.seeds,
            "model_modes": args.model_modes,
            "heterogeneous_architectures": {"resnet18": 0.5, "resnet34": 0.3, "wrn28_10": 0.2},
            "wrn28_10_fallback": "wide_resnet50_2",
            "npz_format": "(K, N_pub, C)",
            "download": False,
            "run_id": run_id,
        },
    )
    print(npz_path)
    print(meta_path)


if __name__ == "__main__":
    main()
