"""MNIST: загрузка, стратифицированное разбиение train/val, DataLoader'ы.

Модуль отвечает за данные и только за них:

1. Скачивает MNIST в ``data/`` (через torchvision).
2. Делит исходный train (60 000) на train и val так, чтобы доли классов
   сохранялись (стратификация). Test остаётся нетронутым — 10 000 объектов.
3. Кэширует разбиение в JSON, чтобы при повторных запусках индексы были
   ровно те же (это проверяется по fingerprint).
4. При необходимости урезает train до подмножества (``train_subset`` для
   этапа 2 и ``limit_train`` для smoke-тестов).
5. Собирает DataLoader'ы с корректным seed и pin_memory для CUDA.

Нормировка — стандартные для MNIST mean/std по всему обучающему набору.
"""

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

# Среднее и стандартное отклонение пикселей MNIST (посчитаны по train).
# Они нужны, чтобы входные значения были примерно в диапазоне [-1, 1],
# иначе обучение с фиксированным lr идёт нестабильно.
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


@dataclass(frozen=True)
class Split:
    """Индексы исходного train-набора, отведённые в train и val.

    Хранятся именно индексы (позиции в MNIST train), а не сами картинки:
    так разбиение занимает килобайты и легко сравнивается по хэшу.
    ``frozen=True`` запрещает случайно перезаписать разбиение после создания.
    """

    train: np.ndarray
    val: np.ndarray

    @property
    def fingerprint(self) -> str:
        """SHA-1 от объединённых индексов — короткий «отпечаток» разбиения.

        Если два запуска дают одинаковый fingerprint, они обучались и
        валидировались на одних и тех же объектах. Значение попадает в
        метаданные и отчёты.
        """
        payload = np.concatenate([self.train, self.val]).astype(np.int64).tobytes()
        return hashlib.sha1(payload).hexdigest()


class IndexedDataset(Dataset):
    """Обёртка над MNIST, возвращающая (image, label, original_index).

    Обычный ``datasets.MNIST`` отдаёт только пару (image, label). Третий
    элемент — индекс объекта в исходном наборе — нужен, чтобы в отчёте
    показать, какие именно картинки модель перепутала
    (см. ``top_errors.csv`` в analysis.py).

    ``base`` — исходный датасет MNIST (train или test), ``indices`` — какие
    его позиции включает эта выборка.
    """

    def __init__(self, base: Dataset, indices: np.ndarray) -> None:
        """Запоминает базовый датасет и список используемых индексов."""
        self.base = base
        # Приводим к int64 сразу: дальше индексы используются для numpy и torch.
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        """Количество объектов в выборке (а не в исходном MNIST)."""
        return int(self.indices.size)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, int, int]:
        """Возвращает (нормализованное изображение, метка, исходный индекс).

        ``position`` — позиция внутри этой выборки, ``index`` — позиция в
        исходном MNIST. Метка приводится к обычному int, чтобы DataLoader
        корректно собрал их в батч.
        """
        index = int(self.indices[position])
        image, label = self.base[index]
        return image, int(label), index


def make_transform() -> transforms.Compose:
    """Создаёт преобразование картинки MNIST: тензор + нормализация.

    ``ToTensor`` переводит PIL-картинку 28×28 в torch.Tensor [1, 28, 28] и
    масштабирует пиксели в [0, 1]; ``Normalize`` вычитает mean и делит на std.
    Один и тот же transform применяется к train, val и test — это важно,
    иначе распределения входов различались бы.
    """
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

    Алгоритм:
      1. Для каждого класса берём floor(размер класса * val_ratio) объектов.
      2. Считаем, сколько объектов не хватает до round(N * val_ratio).
      3. Раздаём остаток классам с наибольшей дробной частью (метод
         наибольших остатков). Благодаря этому итоговый val имеет ровно
         6 000 объектов, а не «плавает» из-за округления по 10 классам.

    Внутри каждого класса порядок перемешивается генератором с seed, поэтому
    разбиение воспроизводимо. Возвращаются отсортированные индексы — так
    проще сравнивать разбиения и кэшировать их.
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio должен быть в (0, 1), получено {val_ratio}")
    rng = np.random.default_rng(seed)
    labels = np.unique(targets)
    # Базовый размер val для каждого класса (округление вниз).
    val_sizes = {
        int(label): int(np.floor(np.count_nonzero(targets == label) * val_ratio))
        for label in labels
    }
    target_val = int(round(targets.size * val_ratio))
    deficit = target_val - sum(val_sizes.values())
    if deficit > 0:
        # Сортируем классы по убыванию дробной части: кто «потерял» больше
        # всего при округлении вниз, тот получает дополнительный объект.
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
        # shuffle на месте; seed общий, поэтому весь процесс воспроизводим.
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
    """Проверяет корректность разбиения и падает при утечке данных.

    Три проверки:
      * train и val не пересекаются (иначе модель видела val при обучении);
      * вместе они покрывают весь исходный набор;
      * внутри объединения нет дубликатов.

    Вызывается и при создании разбиения, и при чтении из кэша: кэш мог быть
    испорчен или заменён вручную.
    """
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
    """Возвращает разбиение из кэша или создаёт и сохраняет новое.

    Кэш — JSON с индексами и метаданными (val_ratio, seed, размеры,
    fingerprint). Если файл уже есть, он читается и проверяется
    ``_verify_split``; иначе разбиение строится заново и записывается.

    Кэширование гарантирует, что все эксперименты (E1–E5) используют одно и
    то же разбиение, даже если запускаются в разные дни.
    """
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
    """Случайно выбирает не более ``limit`` индексов без повторов.

    Используется для трёх вещей:
      * ``train_subset`` — 20 000 объектов на этапе 2;
      * ``limit_train``/``limit_val`` — маленькие выборки в smoke-тестах;
      * разных seed'ов, чтобы каждый прогон E2/E3 брал своё подмножество.

    ``limit=None`` или limit >= размера означает «взять всё как есть».
    Результат сортируется, чтобы порядок не зависел от перемешивания.
    """
    if limit is None or limit >= indices.size:
        return indices
    rng = np.random.default_rng(seed)
    chosen = rng.choice(indices, size=int(limit), replace=False)
    return np.sort(chosen)


def build_datasets(
        config: dict[str, Any],
) -> tuple[IndexedDataset, IndexedDataset, IndexedDataset, dict[str, Any]]:
    """Создаёт train/val/test датасеты и метаданные разбиения.

    Читает секцию ``config["data"]``:
      * ``root`` — куда скачивать MNIST (по умолчанию data/);
      * ``val_ratio`` и ``split_seed`` — параметры разбиения;
      * ``train_subset`` — если задан, train урезается до этого размера
        (20 000 на этапе 2); подмножество зависит от ``config["seed"]``,
        поэтому разные seed'ы видят разные объекты;
      * ``limit_train``/``limit_val`` — ещё более жёсткое урезание для
        smoke-тестов.

    Возвращает три датасета и словарь meta, который затем попадает в
    summary.json/report_data.json: размеры, fingerprint разбиения, хэш
    обучающего подмножества. По этим полям преподаватель может проверить,
    что все эксперименты честно использовали одни и те же val/test.
    """
    data_cfg = config["data"]
    root = project_path(data_cfg.get("root", "data"))
    transform = make_transform()

    # download=True безопасен: если файлы уже есть, torchvision их не качает.
    base_train = datasets.MNIST(
        root=str(root), train=True, download=True, transform=transform
    )
    base_test = datasets.MNIST(
        root=str(root), train=False, download=True, transform=transform
    )

    targets = np.asarray(base_train.targets, dtype=np.int64)
    # Имя кэша включает параметры разбиения: разные val_ratio/seed не
    # перезаписывают друг друга.
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
    subset_size = data_cfg.get("train_subset")
    train_pool = split.train
    if subset_size is not None:
        if int(subset_size) >= split.train.size:
            raise ValueError(
                f"train_subset={subset_size} должен быть меньше train-пула "
                f"({split.train.size})"
            )
        # +1 отделяет seed подмножества от seed инициализации модели,
        # чтобы подвыборка не коррелировала с порядком обучения.
        train_pool = _subset_indices(split.train, int(subset_size), seed + 1)
    train_indices = _subset_indices(
        train_pool, data_cfg.get("limit_train"), seed + 1
    )
    val_indices = _subset_indices(split.val, data_cfg.get("limit_val"), seed + 2)

    # Явные проверки: подмножество train не должно пересекаться с val и
    # не должно содержать дубликатов.
    if np.intersect1d(train_indices, split.val).size:
        raise RuntimeError("Пересечение обучающего подмножества и val — утечка данных")
    if np.unique(train_indices).size != train_indices.size:
        raise RuntimeError("Дубликаты в обучающем подмножестве")

    train_ds = IndexedDataset(base_train, train_indices)
    val_ds = IndexedDataset(base_train, val_indices)
    # Test всегда полный: 10 000 объектов, индексы идут подряд.
    test_ds = IndexedDataset(base_test, np.arange(len(base_test), dtype=np.int64))

    # Хэш фактически использованных train-индексов. У E2/E3 он разный для
    # каждого seed'а — это доказательство, что подмножества действительно
    # случайные, а не одно и то же.
    subset_hash = hashlib.sha1(
        train_indices.astype(np.int64).tobytes()
    ).hexdigest()
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
        "train_subset": None if subset_size is None else int(subset_size),
        "train_subset_seed": seed + 1 if subset_size is not None else None,
        "train_subset_hash": subset_hash,
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
    """Создаёт DataLoader с воспроизводимым перемешиванием.

    Параметры:
        dataset: любая выборка (обычно IndexedDataset).
        batch_size: размер батча.
        shuffle: перемешивать ли порядок (True для train, False для val/test).
        num_workers: число процессов-загрузчиков (0 — всё в основном процессе).
        device: нужен только чтобы включить pin_memory на CUDA.
        seed: seed генератора перемешивания; фиксируется, чтобы эпохи и
            прогоны с одним seed'ом шли в одинаковом порядке.

    ``drop_last=False``: последний неполный батч сохраняется, иначе часть
    объектов терялась бы при подсчёте метрик.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        # pin_memory ускоряет копирование CPU -> GPU.
        "pin_memory": device.type == "cuda",
        "drop_last": False,
        "generator": generator,
    }
    if num_workers > 0:
        # Держим воркеры живыми между эпохами и заранее готовим 2 батча.
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def class_distribution(dataset: IndexedDataset) -> dict[int, int]:
    """Быстрый подсчёт классов без чтения изображений.

    Для MNIST метки лежат в ``dataset.base.targets``, поэтому достаточно
    взять их по индексам выборки — это мгновенно даже для 54 000 объектов.
    Если у базового датасета нет targets, используется медленный путь через
    ``dataset[i][1]``.

    Результат — словарь {цифра: количество}; он пишется в summary.json и
    используется для проверки стратификации.
    """
    targets = getattr(dataset.base, "targets", None)
    if targets is not None:
        labels = np.asarray(targets, dtype=np.int64)[dataset.indices]
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
    values, counts = np.unique(labels, return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts)}
