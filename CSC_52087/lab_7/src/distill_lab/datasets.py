import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T

# pin_memory and num_workers only help on CUDA; they add overhead on MPS/CPU
_USE_CUDA = torch.cuda.is_available()
_LOADER_KWARGS = {"num_workers": 2, "pin_memory": True} if _USE_CUDA else {}


# --- CIFAR-10 ---

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def get_cifar10_loaders(batch_size=128):
    """Return (train_loader, test_loader) for CIFAR-10 with standard augmentation."""
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root="data", train=True, download=True, transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root="data", train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, **_LOADER_KWARGS)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, **_LOADER_KWARGS)
    return train_loader, test_loader


def check_cifar10():
    """Check if CIFAR-10 can be loaded (triggers download if needed)."""
    try:
        torchvision.datasets.CIFAR10(root="data", train=True, download=True)
        return True
    except Exception:
        return False


# --- DermaMNIST ---

DERMAMNIST_CLASSES = [
    "actinic keratosis", "basal cell carcinoma", "benign keratosis",
    "dermatofibroma", "melanoma", "melanocytic nevus", "vascular lesion",
]


class DermaMNISTWrapper(Dataset):
    """Wrapper around MedMNIST DermaMNIST that squeezes labels from (1,) to scalar.

    This is critical because CrossEntropyLoss expects scalar targets, but
    MedMNIST returns labels with shape (1,).
    """

    def __init__(self, split, transform=None, size=64):
        import medmnist
        self.dataset = medmnist.DermaMNIST(
            split=split, transform=transform, download=True,
            size=size, as_rgb=True, root="data",
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, int(label.squeeze())


def get_dermamnist_loaders(batch_size=64):
    """Return (train_loader, val_loader, test_loader) for DermaMNIST 64x64."""
    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(20),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ])
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ])

    train_set = DermaMNISTWrapper("train", transform=train_transform, size=64)
    val_set = DermaMNISTWrapper("val", transform=test_transform, size=64)
    test_set = DermaMNISTWrapper("test", transform=test_transform, size=64)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, **_LOADER_KWARGS)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, **_LOADER_KWARGS)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, **_LOADER_KWARGS)
    return train_loader, val_loader, test_loader


def check_dermamnist():
    """Check if DermaMNIST can be loaded (triggers download if needed)."""
    try:
        import medmnist
        medmnist.DermaMNIST(split="train", download=True, size=64, as_rgb=True, root="data")
        return True
    except Exception:
        return False


def get_class_distribution_fast(dataset):
    """Get class distribution from a dataset using underlying numpy arrays.

    Works with both torchvision datasets (via .targets) and DermaMNISTWrapper
    (via .dataset.labels).
    """
    if hasattr(dataset, "targets"):
        # torchvision datasets (CIFAR-10, etc.)
        labels = np.array(dataset.targets)
    elif hasattr(dataset, "dataset") and hasattr(dataset.dataset, "labels"):
        # DermaMNISTWrapper
        labels = dataset.dataset.labels.squeeze()
    else:
        raise ValueError("Cannot extract labels from this dataset type")

    classes, counts = np.unique(labels, return_counts=True)
    return dict(zip(classes.tolist(), counts.tolist()))
