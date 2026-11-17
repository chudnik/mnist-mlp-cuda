"""Цикл обучения MLP на MNIST с CUDA, AMP, early stopping и чекпоинтами.

Главная функция модуля — ``train()``. Она:
  1. выбирает устройство и фиксирует seed;
  2. собирает train/val датасеты и DataLoader'ы;
  3. создаёт модель строго по конфигу (один скрытый слой);
  4. обучает с CrossEntropy, AdamW и (опционально) AMP/fp16-scaler;
  5. каждую эпоху пишет строку в metrics.csv;
  6. сохраняет best.pt (лучший val loss) и last.pt;
  7. останавливается по early stopping;
  8. сохраняет summary.json и рисует curves.png.

Test-выборка здесь не используется: она нужна только для финальной оценки
в report.py, чтобы не подбирать гиперпараметры по тесту.
"""

from __future__ import annotations

import copy
import csv
import time
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .data import build_datasets, build_loader, class_distribution
from .analysis import run_inference
from .model import build_model
from .utils import (
    count_parameters,
    describe_device,
    get_device,
    project_path,
    save_config,
    save_json,
    set_seed,
    timestamp,
)

# Колонки metrics.csv. Порядок фиксирован, чтобы файл было удобно читать
# pandas'ом и сравнивать между запусками.
CSV_COLUMNS = [
    "epoch",
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
    "lr",
    "seconds",
    "gpu_mem_mb",
    "tanh_saturated",
]


def _activation_saturation(
        model: nn.Module, images: torch.Tensor
) -> float | None:
    """Доля насыщенных tanh-нейронов (|a| > 0.99) на одном батче.

    Зачем это нужно: у tanh производная при |a| -> 1 стремится к нулю, и
    нейроны перестают обучаться («насыщение»). Метрика показывает, какая
    доля выходов tanh уже упёрлась в ±1. Если она близка к 100 %, сеть
    фактически не учится.

    Реализация: временно вешаем forward hook на каждый слой ``nn.Tanh``,
    делаем один проход и усредняем долю насыщенных значений по всем слоям
    активации. Для ReLU-модели возвращается None — там такого понятия нет.

    Важно: модель на время замера переводится в eval и потом возвращается в
    исходный режим, чтобы BatchNorm/Dropout не влияли на замер.
    """
    values: list[float] = []
    handles = []

    def hook(_module: nn.Module, _inputs: Any, output: torch.Tensor) -> None:
        """Считает долю элементов выхода tanh с модулем больше 0.99."""
        values.append(float((output.detach().abs() > 0.99).float().mean()))

    for module in model.modules():
        if isinstance(module, nn.Tanh):
            handles.append(module.register_forward_hook(hook))
    if not handles:
        return None
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(images)
    model.train(was_training)
    # Хуки обязательно снимаем, иначе они накопятся при следующем вызове.
    for handle in handles:
        handle.remove()
    return float(sum(values) / len(values)) if values else None


def _train_epoch(
        model: nn.Module,
        loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        amp: bool,
        amp_dtype: torch.dtype,
        scaler: torch.amp.GradScaler | None,
        grad_clip: float | None,
) -> tuple[float, float]:
    """Одна эпоха обучения; возвращает (средний loss, accuracy).

    Пошагово для каждого батча:
      1. переносит изображения и метки на устройство;
      2. обнуляет градиенты (set_to_none=True — быстрее и экономнее);
      3. прямой проход и loss внутри autocast (если AMP включён);
      4. обратный проход: со scaler'ом для fp16 или обычный для bf16/fp32;
      5. опционально обрезает норму градиента (grad_clip);
      6. шаг оптимизатора.

    Loss и accuracy накапливаются взвешенно по размеру батча, поэтому
    последний неполный батч не искажает среднее. Возвращаемые значения
    считаются по всем объектам эпохи, а не как среднее по батчам.
    """
    # train() включает Dropout и режим обучения BatchNorm.
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _ in loader:
        # Третий элемент (исходный индекс) при обучении не нужен.
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp
        ):
            logits = model(images)
            loss = criterion(logits, labels)

        if scaler is not None and scaler.is_enabled():
            # fp16 требует масштабирования loss, чтобы мелкие градиенты не
            # превращались в ноль. unscale_ нужен перед clip_grad_norm_.
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        # Домножаем на размер батча, чтобы в конце поделить на total.
        total_loss += float(loss.detach()) * labels.size(0)
        correct += int((logits.argmax(dim=1) == labels).sum())
        total += labels.size(0)

    return total_loss / total, correct / total


def _validation(
        model: nn.Module,
        loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        device: torch.device,
        amp: bool,
        amp_dtype: torch.dtype,
) -> tuple[float, float]:
    """Считает loss и accuracy на validation без изменения весов.

    Использует ``run_inference`` (режим eval, без графа), поэтому Dropout
    выключен, а BatchNorm работает по накопленной статистике. Loss считается
    по логитам через ``cross_entropy``, accuracy — по argmax.

    Именно val loss управляет early stopping и выбором best.pt, а не test:
    тест должен оставаться независимым до финального отчёта.
    """
    output = run_inference(
        model, loader, device, amp=amp, amp_dtype=amp_dtype
    )
    loss = float(F.cross_entropy(output["logits"], output["labels"]))
    accuracy = float(
        (output["logits"].argmax(dim=1) == output["labels"]).float().mean()
    )
    return loss, accuracy


def _plot_history(history: list[dict[str, Any]], path: Path) -> None:
    """Рисует кривые loss и accuracy по эпохам и сохраняет PNG.

    Слева — train/val loss, справа — train/val accuracy в процентах.
    График позволяет увидеть переобучение: train loss падает, val loss
    начинает расти. Используется Agg-бэкенд matplotlib (без дисплея).
    """
    if not history:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="val")
    axes[0].set_xlabel("Эпоха")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Потери")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, [row["train_acc"] * 100 for row in history], label="train")
    axes[1].plot(epochs, [row["val_acc"] * 100 for row in history], label="val")
    axes[1].set_xlabel("Эпоха")
    axes[1].set_ylabel("Accuracy, %")
    axes[1].set_title("Точность")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def train(
        config: dict[str, Any],
        name: str = "base",
        verbose: bool = True,
) -> Path:
    """Обучает модель по конфигу и возвращает каталог запуска.

    Аргументы:
        config: полный словарь конфига (секции data/model/training/output).
        name: префикс имени каталога запуска, например "relu" или "tanh".
        verbose: печатать ли прогресс; False используется в массовых
            прогонах (поиск ширины), чтобы не засорять вывод.

    Что сохраняется в ``runs/<name>_<timestamp>/``:
        config.yaml    — точная копия конфига (воспроизводимость);
        data_split.json— метаданные разбиения и подмножества;
        metrics.csv    — по строке на эпоху;
        best.pt        — веса эпохи с минимальным val loss;
        last.pt        — веса последней эпохи;
        curves.png     — графики обучения;
        summary.json   — итог: лучшая эпоха, метрики, время, параметры.

    Почему лучший чекпоинт выбирается по val loss, а не по val accuracy:
    loss — более гладкая величина, accuracy на 6 000 объектов дискретна
    (шаг 1/6000 ≈ 0.017 %) и чаще даёт «плато» с случайными колебаниями.

    Early stopping: если val loss не улучшается ``patience`` эпох подряд
    (с учётом ``min_delta``), обучение останавливается.
    """
    # Если verbose=False, log становится функцией, которая ничего не делает.
    log = print if verbose else (lambda *args, **kwargs: None)
    device = get_device(config.get("device", "auto"))
    seed = int(config.get("seed", 42))
    set_seed(seed)
    if device.type == "cuda":
        # benchmark подбирает самые быстрые свёрточные алгоритмы (для Linear
        # почти не влияет), TF32 ускоряет matmul на Ampere+.
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_cfg = config["training"]
    data_cfg = config["data"]

    log(f"Устройство: {describe_device(device)}")

    train_ds, val_ds, _, meta = build_datasets(config)
    train_loader = build_loader(
        train_ds,
        batch_size=int(data_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(data_cfg["num_workers"]),
        device=device,
        seed=seed,
    )
    val_loader = build_loader(
        val_ds,
        batch_size=int(data_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(data_cfg["num_workers"]),
        device=device,
        seed=seed,
    )

    log(
        f"Данные: train {meta['train_size']}, val {meta['val_size']}, "
        f"test {meta['test_size']} (val_ratio={meta['val_ratio']})"
    )
    # Распределение классов печатается, чтобы сразу видеть стратификацию.
    log(f"Классы train: {class_distribution(train_ds)}")
    log(f"Классы val:   {class_distribution(val_ds)}")

    model = build_model(config["model"]).to(device)
    log(
        f"Модель: 784-{config['model']['hidden_neurons']}-10 "
        f"{config['model']['activation']}, "
        f"dropout={config['model']['dropout']}, "
        f"параметров={count_parameters(model):,}"
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    # Планировщик необязателен: включается только при scheduler: plateau.
    # Он уменьшает lr, когда val loss перестаёт падать.
    scheduler = None
    if train_cfg.get("scheduler") == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(train_cfg.get("scheduler_factor", 0.5)),
            patience=int(train_cfg.get("scheduler_patience", 2)),
            min_lr=1e-6,
        )

    amp_dtype = (
        torch.float16
        if train_cfg.get("amp_dtype", "bf16") == "fp16"
        else torch.bfloat16
    )
    # AMP имеет смысл только на CUDA; на CPU всегда обычный float32.
    amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = None
    if amp and amp_dtype == torch.float16:
        # bf16 не требует scaler'а, fp16 — требует.
        scaler = torch.amp.GradScaler("cuda", enabled=True)

    # Каталог запуска уникален благодаря timestamp.
    run_dir = project_path(config["output"]["dir"]) / f"{name}_{timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, run_dir / "config.yaml")
    save_json(meta, run_dir / "data_split.json")

    metrics_path = run_dir / "metrics.csv"
    history: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    bad_epochs = 0
    patience = int(train_cfg.get("patience", 5))
    # min_delta — минимальное улучшение, которое считается реальным.
    min_delta = float(train_cfg.get("min_delta", 0.0))
    epochs = int(train_cfg["epochs"])
    started = time.time()

    # metrics.csv открыт на запись всё обучение и сбрасывается на диск
    # после каждой эпохи: если процесс упадёт, прогресс сохранится.
    with metrics_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for epoch in range(1, epochs + 1):
            epoch_started = time.time()
            train_loss, train_acc = _train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                amp,
                amp_dtype,
                scaler,
                train_cfg.get("grad_clip"),
            )
            val_loss, val_acc = _validation(
                model, val_loader, criterion, device, amp, amp_dtype
            )
            if scheduler is not None:
                scheduler.step(val_loss)
            lr = float(optimizer.param_groups[0]["lr"])
            seconds = time.time() - epoch_started

            # Диагностика насыщения tanh — только для tanh-моделей.
            saturation = None
            if config["model"]["activation"] == "tanh":
                preview, _, _ = next(iter(val_loader))
                saturation = _activation_saturation(
                    model, preview[:256].to(device)
                )

            # Пиковая память GPU с начала процесса — попадает в отчёт.
            gpu_mem_mb = (
                torch.cuda.max_memory_allocated(device) / 1024 ** 2
                if device.type == "cuda"
                else 0.0
            )

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": lr,
                "seconds": seconds,
                "gpu_mem_mb": gpu_mem_mb,
                # Пустая строка вместо None: CSV не умеет хранить None.
                "tanh_saturated": "" if saturation is None else saturation,
            }
            writer.writerow(row)
            csv_file.flush()
            history.append(row)

            log(
                f"эпоха {epoch:2d}/{epochs} | "
                f"train loss {train_loss:.4f} acc {train_acc * 100:5.2f}% | "
                f"val loss {val_loss:.4f} acc {val_acc * 100:5.2f}% | "
                f"lr {lr:.2e} | {seconds:.1f} с | "
                f"GPU {gpu_mem_mb:.0f} МБ"
                + (
                    f" | tanh sat {saturation * 100:.1f}%"
                    if saturation is not None
                    else ""
                )
            )

            # Улучшение фиксируется только если val loss упал больше чем
            # на min_delta; тогда же обновляется best.pt.
            improved = val_loss < best_val_loss - min_delta
            if improved:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_epoch = epoch
                bad_epochs = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "config": copy.deepcopy(config),
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "data_split": meta,
                        "params": count_parameters(model),
                    },
                    run_dir / "best.pt",
                )
            else:
                bad_epochs += 1

            # last.pt перезаписывается каждую эпоху — полезно, если нужно
            # посмотреть, что было после early stopping.
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": copy.deepcopy(config),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "data_split": meta,
                    "params": count_parameters(model),
                },
                run_dir / "last.pt",
            )

            if bad_epochs >= patience:
                log(
                    f"Early stopping: val loss не улучшается {bad_epochs} эпох"
                )
                break

    elapsed = time.time() - started
    _plot_history(history, run_dir / "curves.png")

    # summary.json — компактный итог запуска для сводных таблиц.
    summary = {
        "run_dir": str(run_dir),
        "device": describe_device(device),
        "params": count_parameters(model),
        "epochs_ran": len(history),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        # True, если обучение прервалось раньше запланированного числа эпох.
        "early_stopped": len(history) < epochs,
        "seconds": elapsed,
        "seconds_per_epoch": elapsed / max(len(history), 1),
        "data": meta,
        "model": config["model"],
        "training": train_cfg,
    }
    save_json(summary, run_dir / "summary.json")

    log(
        f"\nГотово за {elapsed:.1f} с. Лучшая эпоха {best_epoch}: "
        f"val loss {best_val_loss:.4f}, val acc {best_val_acc * 100:.2f}%"
    )
    log(f"Чекпоинт: {run_dir / 'best.pt'}")
    log(f"Логи:     {run_dir / 'metrics.csv'}")
    return run_dir
