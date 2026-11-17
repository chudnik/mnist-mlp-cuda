"""Оценка обученной модели и сбор данных для отчёта.

Модуль умеет:
  * загружать чекпоинт (best.pt) вместе с его конфигом;
  * прогонять модель по выборке и собирать логиты/вероятности;
  * считать accuracy, macro/weighted F1, матрицу ошибок и метрики по классам;
  * сохранять report.txt, confusion_matrix.png, per_class.csv и top_errors.csv;
  * быстро возвращать accuracy на одной выборке (``accuracy_on_split``) —
    это используется при поиске ширины и в сводных таблицах.

Ничего не обучает: только inference в режиме ``model.eval()``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from .data import IndexedDataset, build_datasets, build_loader
from .model import MLP, build_model
from .utils import describe_device, get_device, project_path, save_json

# Имена классов для отчётов: просто цифры 0..9 в виде строк.
CLASS_NAMES = [str(digit) for digit in range(10)]


def load_checkpoint(
        path: str | Path, device: torch.device
) -> tuple[MLP, dict[str, Any]]:
    """Загружает модель из чекпоинта и переводит её в eval-режим.

    Чекпоинт — словарь, который сохраняет train.py:
      * ``config`` — полный конфиг запуска (по нему восстанавливается
        архитектура: число нейронов, активация, dropout и т.д.);
      * ``model_state`` — веса;
      * ``epoch``, ``val_loss``, ``val_acc`` — метаданные лучшей эпохи.

    ``map_location="cpu"`` позволяет читать файл даже без GPU; затем модель
    переносится на нужное устройство. ``eval()`` обязательно: он выключает
    Dropout и переводит BatchNorm в режим использования накопленной статистики.
    """
    checkpoint = torch.load(project_path(path), map_location="cpu", weights_only=False)
    model = build_model(checkpoint["config"]["model"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint


@torch.no_grad()
def run_inference(
        model: torch.nn.Module,
        loader: DataLoader,
        device: torch.device,
        amp: bool = False,
        amp_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    """Прогоняет модель по DataLoader'у и собирает результаты.

    Возвращает словарь:
      * ``logits`` — сырые выходы последнего слоя, форма (N, 10);
      * ``probs`` — softmax по логитам, форма (N, 10);
      * ``labels`` — истинные метки, форма (N,);
      * ``indices`` — индексы объектов в исходном MNIST, форма (N,).

    ``@torch.no_grad()`` отключает построение графа: инференс быстрее и
    требует меньше памяти. AMP (``autocast``) включается только на CUDA;
    результаты всё равно приводятся к float32, чтобы метрики считались
    одинаково на CPU и GPU.

    Данные накапливаются по батчам в списках и склеиваются один раз в конце —
    так не нужно заранее знать размер выборки.
    """
    model.eval()
    logits_parts: list[torch.Tensor] = []
    labels_parts: list[torch.Tensor] = []
    index_parts: list[torch.Tensor] = []

    use_amp = amp and device.type == "cuda"
    for images, labels, indices in loader:
        # non_blocking позволяет копировать батч асинхронно с pinned memory.
        images = images.to(device, non_blocking=True)
        with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=use_amp
        ):
            logits = model(images)
        # Накопление всегда в float32 и на CPU: экономим видеопамять и
        # получаем одинаковые числа независимо от устройства.
        logits_parts.append(logits.float().cpu())
        labels_parts.append(labels)
        index_parts.append(indices)

    logits = torch.cat(logits_parts)
    labels = torch.cat(labels_parts)
    indices = torch.cat(index_parts)
    return {
        "logits": logits,
        "probs": torch.softmax(logits, dim=1),
        "labels": labels,
        "indices": indices,
    }


def compute_metrics(labels: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    """Считает основные метрики классификации по меткам и вероятностям.

    Вход:
        labels: истинные метки, массив (N,);
        probs: вероятности классов, массив (N, 10).

    Выход — словарь:
        accuracy, macro_f1, weighted_f1 — общие метрики;
        per_class — precision/recall/f1/support для каждой цифры;
        confusion_matrix — матрица 10×10 (строки — истина, столбцы — прогноз);
        num_samples — размер выборки.

    Macro F1 считает каждую цифру равноправно, weighted F1 — с учётом
    количества объектов класса. ``zero_division=0`` защищает от деления на
    ноль, если какой-то класс вообще не встретился.
    """
    # argmax по вероятностям — итоговый прогноз класса.
    predictions = probs.argmax(axis=1)
    accuracy = float((predictions == labels).mean())
    matrix = confusion_matrix(labels, predictions, labels=list(range(10)))
    report = classification_report(
        labels,
        predictions,
        labels=list(range(10)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    per_class = {
        digit: {
            "precision": float(report[digit]["precision"]),
            "recall": float(report[digit]["recall"]),
            "f1": float(report[digit]["f1-score"]),
            "support": int(report[digit]["support"]),
        }
        for digit in CLASS_NAMES
    }
    return {
        "accuracy": accuracy,
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "num_samples": int(labels.size),
    }


def plot_confusion(matrix: np.ndarray, path: str | Path, title: str) -> None:
    """Рисует матрицу ошибок 10×10 и сохраняет её в PNG.

    В каждой клетке — количество объектов. Цвет текста выбирается по
    яркости фона (белый на тёмных клетках, чёрный на светлых), чтобы числа
    читались. ``matplotlib.use("Agg")`` нужен для работы без дисплея
    (сервер/CI). Импорт matplotlib сделан внутри функции, чтобы модуль
    analysis можно было использовать без установленного matplotlib.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xlabel("Предсказание")
    ax.set_ylabel("Истинный класс")
    ax.set_title(title)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    # Порог для смены цвета подписи: половина максимального значения.
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if matrix[row, column] > threshold else "black",
            )
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    # Явно закрываем фигуру, иначе при массовых прогонах копится память.
    plt.close(fig)


def _select_dataset(
        split: str,
        datasets: tuple[IndexedDataset, IndexedDataset, IndexedDataset],
) -> IndexedDataset:
    """Выбирает выборку по имени: "train", "val" или "test".

    ``datasets`` — результат ``build_datasets()`` в порядке
    (train, val, test). Отдельная функция нужна, чтобы не дублировать
    if/elif в evaluate_split и accuracy_on_split.
    """
    if split == "train":
        return datasets[0]
    if split == "val":
        return datasets[1]
    if split == "test":
        return datasets[2]
    raise ValueError(f"Неизвестная выборка: {split!r}; ожидается train/val/test")


def evaluate_split(
        checkpoint_path: str | Path,
        split: str,
        out_dir: str | Path,
        top_errors: int = 100,
) -> dict[str, Any]:
    """Считает метрики на выборке и сохраняет артефакты в out_dir.

    Что делает по шагам:
      1. Загружает чекпоинт и восстанавливает по его конфигу данные.
      2. Собирает нужную выборку (train/val/test) и DataLoader без shuffle.
      3. Прогоняет inference и считает метрики.
      4. Сохраняет report.txt, confusion_matrix.png, per_class.csv,
         top_errors.csv и metrics.json.
      5. Возвращает payload — он же попадает в report_data.json.

    ``top_errors`` — сколько самых уверенных ошибок записать в CSV. Для них
    сохраняются исходный индекс MNIST, истинная и предсказанная цифры,
    уверенность модели в ошибке и вероятность правильного класса. Это
    удобно для визуального разбора.

    Важно: limit_train/limit_val из конфига сбрасываются, чтобы отчёт
    всегда считался на полных выборках, даже если обучение было smoke-тестом.
    """
    device = get_device("auto")
    model, checkpoint = load_checkpoint(checkpoint_path, device)

    config = copy.deepcopy(checkpoint["config"])
    # Для отчёта всегда используются полные выборки, без smoke-ограничений.
    config["data"]["limit_train"] = None
    config["data"]["limit_val"] = None

    datasets = build_datasets(config)[:3]
    dataset = _select_dataset(split, datasets)
    loader = build_loader(
        dataset,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
        device=device,
        seed=int(config.get("seed", 42)),
    )

    result = run_inference(
        model,
        loader,
        device,
        amp=bool(config["training"].get("amp", True)),
        # В конфиге хранится строка "bf16"/"fp16" — переводим в torch.dtype.
        amp_dtype=torch.float16
        if config["training"].get("amp_dtype", "bf16") == "fp16"
        else torch.bfloat16,
    )
    labels = result["labels"].numpy()
    probs = result["probs"].numpy()
    metrics = compute_metrics(labels, probs)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Текстовый отчёт sklearn — привычный формат precision/recall/f1.
    predictions = probs.argmax(axis=1)
    report_text = classification_report(
        labels,
        predictions,
        labels=list(range(10)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    (out_dir / "report.txt").write_text(report_text, encoding="utf-8")

    matrix = np.asarray(metrics["confusion_matrix"])
    plot_confusion(
        matrix, out_dir / "confusion_matrix.png", f"Confusion matrix: {split}"
    )

    # per_class.csv — те же метрики по классам в машиночитаемом виде.
    with (out_dir / "per_class.csv").open("w", encoding="utf-8") as fh:
        fh.write("digit,precision,recall,f1,support\n")
        for digit, values in metrics["per_class"].items():
            fh.write(
                f"{digit},{values['precision']:.6f},{values['recall']:.6f},"
                f"{values['f1']:.6f},{values['support']}\n"
            )

    # Самые уверенные ошибки — полезно смотреть глазами при анализе.
    # wrong — позиции ошибочных объектов, order сортирует их по убыванию
    # вероятности, которую модель дала неправильному классу.
    wrong = np.flatnonzero(predictions != labels)
    order = np.argsort(-probs[wrong, predictions[wrong]])[:top_errors]
    source_indices = result["indices"].numpy()
    with (out_dir / "top_errors.csv").open("w", encoding="utf-8") as fh:
        fh.write(
            "mnist_index,true_label,predicted_label,confidence,true_probability\n"
        )
        for position in wrong[order]:
            fh.write(
                f"{int(source_indices[position])},{int(labels[position])},"
                f"{int(predictions[position])},"
                f"{float(probs[position, predictions[position]]):.6f},"
                f"{float(probs[position, labels[position]]):.6f}\n"
            )

    payload = {
        "checkpoint": str(project_path(checkpoint_path)),
        "split": split,
        # -1/NaN означают, что в чекпоинте не было этих полей.
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_val_loss": float(checkpoint.get("val_loss", float("nan"))),
        "checkpoint_val_acc": float(checkpoint.get("val_acc", float("nan"))),
        "device": describe_device(device),
        **metrics,
    }
    save_json(payload, out_dir / "metrics.json")

    print(
        f"[{split}] accuracy {metrics['accuracy'] * 100:.2f} %, "
        f"macro F1 {metrics['macro_f1'] * 100:.2f} %"
    )
    return payload


def accuracy_on_split(
    checkpoint_path: str | Path,
    split: str = "test",
    batch_size: int = 512,
    device_preference: str = "auto",
) -> float:
    """Быстрая accuracy на выборке без сохранения артефактов.

    Нужна там, где важна только одна цифра:
      * поиск минимальной ширины (run_min_neurons.py) — test accuracy;
      * сводные таблицы (run_summary.py) — train accuracy лучшего чекпоинта.

    Отличие от evaluate_split: не пишет файлы, не рисует графики, не считает
    F1. Поэтому работает заметно быстрее. Batch size и устройство можно
    задать явно; limit_train/limit_val, как и в отчёте, сбрасываются.
    """
    device = get_device(device_preference)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    config = copy.deepcopy(checkpoint["config"])
    config["data"]["limit_train"] = None
    config["data"]["limit_val"] = None
    config["data"]["batch_size"] = batch_size

    datasets = build_datasets(config)[:3]
    dataset = _select_dataset(split, datasets)
    loader = build_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
        device=device,
        seed=int(config.get("seed", 42)),
    )
    result = run_inference(
        model,
        loader,
        device,
        amp=bool(config["training"].get("amp", True)),
        amp_dtype=torch.float16
        if config["training"].get("amp_dtype", "bf16") == "fp16"
        else torch.bfloat16,
    )
    # argmax логитов эквивалентен argmax вероятностей, но без softmax.
    correct = (result["logits"].argmax(dim=1) == result["labels"]).float().mean()
    return float(correct)
