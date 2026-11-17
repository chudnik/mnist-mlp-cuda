"""MNIST: загрузка, стратифицированное разбиение train/val, DataLoader'ы."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from .utils import load_json, project_path, save_json

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    val: np.ndarray

    @property
    def fingerprint(self) -> str:
        payload = np.concatenate([self.train, self.val]).astype(np.int64).tobytes()
        return hashlib.sha1(payload).hexdigest()


class IndexedDataset(Dataset):
    """Обёртка над MNIST, возвращающая (image, label, original_index)."""

    def __init__(self, base: Dataset, indices: np.ndarray) -> None:
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, int, int]:
        index = int(self.indices[position])
        image, label = self.base[index]
        return image, int(label), index


def make_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    )


def stratified_split(
    targets: np.ndarray,
    val_ratio: float,
    seed: int,
) -> Split:
    """Делит индексы по классам, сохраняя доли классов в train и val.

    Размер val — ровно round(N * val_ratio): остатки распределяются по классам
    с наибольшей дробной частью (метод наибольших остатков), поэтому итоговые
    54 000 / 6 000 не «плывут» из-за округления по каждому классу.
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio должен быть в (0, 1), получено {val_ratio}")
    rng = np.random.default_rng(seed)
    labels = np.unique(targets)
    val_sizes = {
        int(label): int(np.floor(np.count_nonzero(targets == label) * val_ratio))
        for label in labels
    }
    target_val = int(round(targets.size * val_ratio))
    deficit = target_val - sum(val_sizes.values())
    if deficit > 0:
        remainders = sorted(
            labels,
            key=lambda label: (
                -(np.count_nonzero(targets == label) * val_ratio
                  - val_sizes[int(label)]),
                int(label),
            ),
        )
        for label in remainders[:deficit]:
            val_sizes[int(label)] += 1

    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for label in labels:
        class_indices = np.flatnonzero(targets == label)
        rng.shuffle(class_indices)
        val_size = val_sizes[int(label)]
        val_parts.append(class_indices[:val_size])
        train_parts.append(class_indices[val_size:])
    split = Split(
        train=np.sort(np.concatenate(train_parts)),
        val=np.sort(np.concatenate(val_parts)),
    )
    _verify_split(split, len(targets))
    return split


def _verify_split(split: Split, total: int) -> None:
    if np.intersect1d(split.train, split.val).size:
        raise RuntimeError("Пересечение train и val — утечка данных")
    union = np.concatenate([split.train, split.val])
    if union.size != total:
        raise RuntimeError(f"Разбиение покрывает {union.size} из {total} объектов")
    if np.unique(union).size != union.size:
        raise RuntimeError("Дубликаты в разбиении")


def load_split(
    targets: np.ndarray,
    cache_path: str | Path,
    val_ratio: float,
    seed: int,
) -> Split:
    cache_path = Path(cache_path)
    if cache_path.exists():
        payload = load_json(cache_path)
        split = Split(
            train=np.asarray(payload["train"], dtype=np.int64),
            val=np.asarray(payload["val"], dtype=np.int64),
        )
        _verify_split(split, len(targets))
        return split

    split = stratified_split(targets, val_ratio, seed)
    save_json(
        {
            "val_ratio": val_ratio,
            "seed": seed,
            "train_size": int(split.train.size),
            "val_size": int(split.val.size),
            "fingerprint": split.fingerprint,
            "train": split.train.tolist(),
            "val": split.val.tolist(),
        },
        cache_path,
    )
    return split


def _subset_indices(
    indices: np.ndarray,
    limit: int | None,
    seed: int,
) -> np.ndarray:
    if limit is None or limit >= indices.size:
        return indices
    rng = np.random.default_rng(seed)
    chosen = rng.choice(indices, size=int(limit), replace=False)
    return np.sort(chosen)


def build_datasets(
    config: dict[str, Any],
) -> tuple[IndexedDataset, IndexedDataset, IndexedDataset, dict[str, Any]]:
    """Возвращает train/val/test датасеты и метаданные разбиения."""
    data_cfg = config["data"]
    root = project_path(data_cfg.get("root", "data"))
    transform = make_transform()

    base_train = datasets.MNIST(
        root=str(root), train=True, download=True, transform=transform
    )
    base_test = datasets.MNIST(
        root=str(root), train=False, download=True, transform=transform
    )

    targets = np.asarray(base_train.targets, dtype=np.int64)
    split_path = root / "splits" / (
        f"val{data_cfg['val_ratio']:.3f}_seed{data_cfg['split_seed']}.json"
    )
    split = load_split(
        targets=targets,
        cache_path=split_path,
        val_ratio=float(data_cfg["val_ratio"]),
        seed=int(data_cfg["split_seed"]),
    )

    seed = int(config.get("seed", 42))
    train_indices = _subset_indices(
        split.train, data_cfg.get("limit_train"), seed + 1
    )
    val_indices = _subset_indices(split.val, data_cfg.get("limit_val"), seed + 2)

    train_ds = IndexedDataset(base_train, train_indices)
    val_ds = IndexedDataset(base_train, val_indices)
    test_ds = IndexedDataset(base_test, np.arange(len(base_test), dtype=np.int64))

    meta = {
        "split_file": str(split_path),
        "fingerprint": split.fingerprint,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "val_ratio": float(data_cfg["val_ratio"]),
        "split_seed": int(data_cfg["split_seed"]),
        "train_full_size": int(split.train.size),
        "val_full_size": int(split.val.size),
    }
    return train_ds, val_ds, test_ds, meta


def build_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
    seed: int = 0,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
        "generator": generator,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def class_distribution(dataset: IndexedDataset) -> dict[int, int]:
    """Быстрый подсчёт классов без чтения изображений."""
    targets = getattr(dataset.base, "targets", None)
    if targets is not None:
        labels = np.asarray(targets, dtype=np.int64)[dataset.indices]
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
    values, counts = np.unique(labels, return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts)}
