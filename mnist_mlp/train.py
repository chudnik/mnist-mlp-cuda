"""Цикл обучения MLP на MNIST с CUDA, AMP, early stopping и чекпоинтами."""

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
    """Доля насыщенных tanh-нейронов (|a| > 0.99) на одном батче."""
    values: list[float] = []
    handles = []

    def hook(_module: nn.Module, _inputs: Any, output: torch.Tensor) -> None:
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
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp
        ):
            logits = model(images)
            loss = criterion(logits, labels)

        if scaler is not None and scaler.is_enabled():
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
    output = run_inference(
        model, loader, device, amp=amp, amp_dtype=amp_dtype
    )
    loss = float(F.cross_entropy(output["logits"], output["labels"]))
    accuracy = float(
        (output["logits"].argmax(dim=1) == output["labels"]).float().mean()
    )
    return loss, accuracy


def _plot_history(history: list[dict[str, Any]], path: Path) -> None:
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
    log = print if verbose else (lambda *args, **kwargs: None)
    device = get_device(config.get("device", "auto"))
    seed = int(config.get("seed", 42))
    set_seed(seed)
    if device.type == "cuda":
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
    amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = None
    if amp and amp_dtype == torch.float16:
        scaler = torch.amp.GradScaler("cuda", enabled=True)

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
    min_delta = float(train_cfg.get("min_delta", 0.0))
    epochs = int(train_cfg["epochs"])
    started = time.time()

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

            saturation = None
            if config["model"]["activation"] == "tanh":
                preview, _, _ = next(iter(val_loader))
                saturation = _activation_saturation(
                    model, preview[:256].to(device)
                )

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

    summary = {
        "run_dir": str(run_dir),
        "device": describe_device(device),
        "params": count_parameters(model),
        "epochs_ran": len(history),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
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
