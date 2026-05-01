from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from torch.utils.data import Dataset
from torchvision import datasets, transforms


def _project_root_candidates() -> List[Path]:
    cands = []
    cur = Path.cwd().resolve()
    for p in [cur, *cur.parents]:
        if (p / "observations").exists() and (p / "analysis").exists():
            cands.append(p)
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "observations").exists() and (p / "analysis").exists():
            cands.append(p)
    out = []
    seen = set()
    for c in cands:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def resolve_data_root(data_root: str) -> Tuple[Path, List[Path]]:
    raw = Path(data_root)
    tried: List[Path] = []

    if raw.is_absolute():
        tried.append(raw)
        if raw.exists():
            print(f"[Data] Using data root: {raw}")
            return raw, tried
    else:
        c1 = (Path.cwd() / raw).resolve()
        tried.append(c1)
        if c1.exists():
            print(f"[Data] Using data root: {c1}")
            return c1, tried

    for root in _project_root_candidates():
        c2 = (root / raw).resolve()
        tried.append(c2)
        if c2.exists():
            print(f"[Data] Using data root: {c2}")
            return c2, tried
        c3 = (root / "analysis" / "data").resolve()
        tried.append(c3)
        if c3.exists():
            print(f"[Data] Using data root: {c3}")
            return c3, tried

    raise FileNotFoundError(
        "Data root not found.\n"
        f"Current working directory: {Path.cwd()}\n"
        "Resolved data root candidates tried:\n  - " + "\n  - ".join(str(x) for x in tried)
    )


def _resolve_dataset_dir(root: Path, name: str) -> Path:
    if root.name == name:
        return root
    return root / name


def _check_cifar10_ready(cifar_dir: Path) -> List[str]:
    required = [
        "data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5", "test_batch", "batches.meta"
    ]
    missing = [f for f in required if not (cifar_dir / f).exists()]
    return missing


def get_dataset(dataset: str, data_root: str) -> Tuple[Dataset, int, int, int, Path]:
    root, tried = resolve_data_root(data_root)
    name = dataset.lower()

    if name == "cifar10":
        cifar_dir = _resolve_dataset_dir(root, "cifar-10-batches-py")
        missing = _check_cifar10_ready(cifar_dir)
        if missing:
            raise FileNotFoundError(
                "CIFAR-10 not found.\n"
                f"Current working directory: {Path.cwd()}\n"
                "Resolved data_root candidates tried:\n  - " + "\n  - ".join(str(x) for x in tried) + "\n"
                f"Resolved data root: {root}\n"
                f"Expected CIFAR-10 directory:\n  - {cifar_dir}\n"
                "Missing required files:\n  - " + "\n  - ".join(missing)
            )
        tv_root = cifar_dir.parent if cifar_dir.name == "cifar-10-batches-py" else root
        tfm = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
        ds = datasets.CIFAR10(root=str(tv_root), train=True, download=False, transform=tfm)
        print(f"[Data] CIFAR-10 found at {cifar_dir}")
        return ds, 10, 3, 32, root

    if name == "mnist":
        mnist_dir = _resolve_dataset_dir(root, "MNIST")
        if not mnist_dir.exists():
            raise FileNotFoundError(
                "MNIST not found.\n"
                f"Current working directory: {Path.cwd()}\n"
                "Resolved data_root candidates tried:\n  - " + "\n  - ".join(str(x) for x in tried) + "\n"
                f"Resolved data root: {root}\n"
                f"Expected MNIST directory:\n  - {mnist_dir}"
            )
        tv_root = mnist_dir.parent if mnist_dir.name == "MNIST" else root
        tfm = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
        ds = datasets.MNIST(root=str(tv_root), train=True, download=False, transform=tfm)
        print(f"[Data] MNIST found at {mnist_dir}")
        return ds, 10, 1, 32, root

    if name == "tinyimagenet":
        tiny_dir = _resolve_dataset_dir(root, "tiny-imagenet-200")
        train_dir = tiny_dir / "train"
        if not train_dir.exists():
            raise FileNotFoundError(
                "Tiny-ImageNet not found.\n"
                f"Current working directory: {Path.cwd()}\n"
                "Resolved data_root candidates tried:\n  - " + "\n  - ".join(str(x) for x in tried) + "\n"
                f"Resolved data root: {root}\n"
                f"Expected Tiny-ImageNet directory:\n  - {tiny_dir}"
            )
        tfm = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
        ds = datasets.ImageFolder(root=str(train_dir), transform=tfm)
        print(f"[Data] Tiny-ImageNet found at {tiny_dir}")
        return ds, 200, 3, 64, root

    raise ValueError(f"Unsupported dataset: {dataset}")
