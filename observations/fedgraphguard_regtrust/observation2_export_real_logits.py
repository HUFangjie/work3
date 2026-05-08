from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

if __package__ in (None, ""):
    import sys
    sys.path.append(str(Path(__file__).resolve().parent))
    from export_real_logits import dirichlet_partition, _collect_labels, _train_one_client, _public_logits
    from obs_datasets import get_dataset
    from obs_model_factory import assign_architectures, build_model
    from obs_utils import resolve_out_dir, save_json
else:
    from .export_real_logits import dirichlet_partition, _collect_labels, _train_one_client, _public_logits
    from .obs_datasets import get_dataset
    from .obs_model_factory import assign_architectures, build_model
    from .obs_utils import resolve_out_dir, save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cifar10", choices=["mnist", "cifar10", "tinyimagenet"])
    p.add_argument("--data-root", default="analysis/data")
    p.add_argument("--out-dir", default="observations/real_inputs/obs2")
    p.add_argument("--num-clients", type=int, default=10)
    p.add_argument("--num-public", type=int, default=1000)
    p.add_argument("--betas", nargs="+", type=float, default=[0.3, 0.5, 0.7, 1.0])
    p.add_argument("--model-modes", nargs="+", default=["homogeneous", "heterogeneous"], choices=["homogeneous", "heterogeneous"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--local-epochs", type=int, default=10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    out_dir = resolve_out_dir(args.out_dir)
    ds, num_classes, in_ch, _, _ = get_dataset(args.dataset, args.data_root)
    labels = _collect_labels(ds)
    device = torch.device(args.device)

    meta = {"files": []}

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
                    subset = Subset(ds, global_parts[cid])
                    bs = min(64, max(2, len(subset)))
                    c_loader = DataLoader(subset, batch_size=bs, shuffle=True, num_workers=2, drop_last=True)
                    _train_one_client(model, c_loader, device, epochs=args.local_epochs)
                    logits_all.append(_public_logits(model, public_loader, device))

                arr = np.stack(logits_all, axis=0)
                name = f"obs2_logits_dataset_{args.dataset}_mode_{mode}_beta_{beta}_seed_{seed}.npy"
                path = out_dir / name
                np.save(path, arr)
                meta["files"].append({
                    "file": str(path), "dataset": args.dataset, "mode": mode, "beta": beta, "seed": seed,
                    "num_clients": args.num_clients, "num_public": args.num_public, "local_epochs": args.local_epochs,
                    "shape": list(arr.shape),
                })
                print(path)

    save_json(out_dir / "obs2_metadata.json", meta)


if __name__ == "__main__":
    main()
