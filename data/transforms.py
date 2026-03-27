# data/transforms.py

from typing import Tuple
from torchvision import transforms


def get_fmnist_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Train/test transforms for Fashion-MNIST (28x28 grayscale)."""
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    return train_transform, test_transform


def get_cifar10_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Train/test transforms for CIFAR-10 (32x32 RGB)."""
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
    ])
    return train_transform, test_transform


def get_tiny_imagenet_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Train/test transforms for Tiny-ImageNet-200 (64x64 RGB)."""
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_transform, test_transform


def get_pathmnist_transforms(image_size: int = 28) -> Tuple[transforms.Compose, transforms.Compose]:
    """Train/test transforms for PathMNIST (28x28 RGB).

    PathMNIST images are already 28x28, but we keep an optional `image_size`
    for robustness (e.g., if you want to run a larger backbone and upsample).
    """
    # Conservative normalization; PathMNIST is histopathology, so ImageNet stats
    # are not necessarily appropriate. Keeping (0.5,0.5,0.5) is a strong baseline.
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    train_ops = []
    test_ops = []
    if image_size != 28:
        train_ops.append(transforms.Resize((image_size, image_size)))
        test_ops.append(transforms.Resize((image_size, image_size)))

    # Light augmentation only (avoid heavy color jitter unless needed).
    train_ops += [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    test_ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    return transforms.Compose(train_ops), transforms.Compose(test_ops)


def get_train_transform(dataset: str):
    dataset = dataset.lower()
    if dataset in ["fmnist", "fashion_mnist"]:
        train_t, _ = get_fmnist_transforms()
        return train_t
    if dataset in ["femnist", "emnist"]:
        train_t, _ = get_fmnist_transforms()
        return train_t
    if dataset == "cifar10":
        train_t, _ = get_cifar10_transforms()
        return train_t
    if dataset in ["tiny_imagenet", "tinyimagenet", "tiny-imagenet"]:
        train_t, _ = get_tiny_imagenet_transforms()
        return train_t
    if dataset in ["pathmnist", "path-mnist"]:
        train_t, _ = get_pathmnist_transforms()
        return train_t
    raise ValueError(f"Unsupported dataset for train transform: {dataset}")


def get_test_transform(dataset: str):
    dataset = dataset.lower()
    if dataset in ["fmnist", "fashion_mnist"]:
        _, test_t = get_fmnist_transforms()
        return test_t
    if dataset in ["femnist", "emnist"]:
        _, test_t = get_fmnist_transforms()
        return test_t
    if dataset == "cifar10":
        _, test_t = get_cifar10_transforms()
        return test_t
    if dataset in ["tiny_imagenet", "tinyimagenet", "tiny-imagenet"]:
        _, test_t = get_tiny_imagenet_transforms()
        return test_t
    if dataset in ["pathmnist", "path-mnist"]:
        _, test_t = get_pathmnist_transforms()
        return test_t
    raise ValueError(f"Unsupported dataset for test transform: {dataset}")
