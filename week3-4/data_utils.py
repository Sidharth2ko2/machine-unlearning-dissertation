"""Data loading for Week 3 — same as Week 2, reuses Week 1 data."""
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

from config import BATCH_SIZE, DATA_DIR, FORGET_CLASS


def get_transforms():
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    return train_tf, test_tf


def get_datasets():
    train_tf, test_tf = get_transforms()
    train_ds = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=False, transform=train_tf)
    test_ds = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=False, transform=test_tf)
    return train_ds, test_ds


def get_all_loaders(batch_size=BATCH_SIZE):
    train_ds, test_ds    = get_datasets()
    targets              = np.array(train_ds.targets)
    test_targets         = np.array(test_ds.targets)

    forget_idx           = np.where(targets      == FORGET_CLASS)[0]
    retain_idx           = np.where(targets      != FORGET_CLASS)[0]
    forget_class_test_idx = np.where(test_targets == FORGET_CLASS)[0]

    return {
        'train':             DataLoader(train_ds,                       batch_size=batch_size, shuffle=True,  num_workers=2),
        'test':              DataLoader(test_ds,                        batch_size=batch_size, shuffle=False, num_workers=2),
        'forget':            DataLoader(Subset(train_ds, forget_idx),   batch_size=batch_size, shuffle=True,  num_workers=2),
        'retain':            DataLoader(Subset(train_ds, retain_idx),   batch_size=batch_size, shuffle=True,  num_workers=2),
        'forget_class_test': DataLoader(Subset(test_ds, forget_class_test_idx), batch_size=batch_size, shuffle=False, num_workers=2),
    }
