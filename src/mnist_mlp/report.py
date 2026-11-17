"""Сборка Markdown-отчёта и сводного JSON из результатов обучения."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import load_json, save_json


def _pct(value: float) -> str:
    return f"{value * 100:.2f} %"


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def build_report(
    run_dir: str | Path,
    results: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> Path:
    run_dir = Path(run_dir)
    summary = load_json(run_dir / "summary.json")
    meta = summary["data"]
    model_cfg = config["model"]
    training_cfg = config["training"]
    test = results["test"]

    gap = results["train"]["accuracy"] - test["accuracy"]
    hidden = "-".join(str(width) for width in model_cfg["hidden"])

    lines: list[str] = [
        "# Отчёт об обучении MLP на MNIST",
        "",
        f"Запуск: `{run_dir.name}`  ",
        f"Устройство: {summary['device']}",
        "",
        "## Данные",
        "",
        f"- train: {meta['train_size']}",
        f"- validation: {meta['val_size']}",
        f"- test: {meta['test_size']}",
        f"- val_ratio: {meta['val_ratio']}, split seed: {meta['split_seed']}",
        f"- fingerprint разбиения: `{meta['fingerprint']}`",
        "",
        "## Модель",
        "",
        f"- архитектура: 784-{hidden}-10",
        f"- активация: {model_cfg['activation']}",
        f"- dropout: {model_cfg['dropout']}, batchnorm: {model_cfg['batchnorm']}",
        f"- параметров: {summary['params']:,}",
        "",
        "## Обучение",
        "",
        f"- эпох выполнено: {summary['epochs_ran']} (лимит {training_cfg['epochs']})",
        f"- лучшая эпоха: {summary['best_epoch']}",
        f"- early stopping: {'да' if summary['early_stopped'] else 'нет'}",
        f"- время: {summary['seconds']:.1f} с "
        f"(≈{summary['seconds_per_epoch']:.2f} с/эпоха)",
        f"- оптимизатор: AdamW, lr={training_cfg['lr']}, "
        f"weight_decay={training_cfg['weight_decay']}",
        f"- AMP: {training_cfg['amp']} ({training_cfg['amp_dtype']}), "
        f"batch={meta.get('batch_size', config['data']['batch_size'])}",
        "",
        "## Результаты",
        "",
    ]

    lines += _table(
        ["Выборка", "Accuracy", "Macro F1", "Weighted F1", "Примеров"],
        [
            [
                split,
                _pct(results[split]["accuracy"]),
                _pct(results[split]["macro_f1"]),
                _pct(results[split]["weighted_f1"]),
                str(results[split]["num_samples"]),
            ]
            for split in ("train", "val", "test")
        ],
    )
    lines += [
        "",
        f"Разрыв train–test: **{gap * 100:.2f} п.п.**  ",
        f"Лучшая val accuracy: **{_pct(summary['best_val_acc'])}** "
        f"(val loss {summary['best_val_loss']:.4f})",
        "",
        "## Качество по классам (test)",
        "",
    ]
    lines += _table(
        ["Цифра", "Precision", "Recall", "F1", "Support"],
        [
            [
                digit,
                _pct(values["precision"]),
                _pct(values["recall"]),
                _pct(values["f1"]),
                str(values["support"]),
            ]
            for digit, values in test["per_class"].items()
        ],
    )
    lines += [
        "",
        "## Артефакты",
        "",
        "- `best.pt`, `last.pt` — чекпоинты",
        "- `metrics.csv` — метрики по эпохам (loss, accuracy, lr, время, GPU)",
        "- `curves.png` — кривые обучения",
        "- `summary.json` — сводка обучения",
        "- `eval_train/`, `eval_val/`, `eval_test/` — метрики, "
        "classification report, confusion matrix, ошибки",
        "- `report_data.json` — все числа одним файлом",
        "",
        "## Примечание",
        "",
        "Test использовался только для финальной оценки выбранного по validation "
        "чекпоинта; гиперпараметры по test не подбирались.",
        "",
    ]

    report_path = run_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    save_json(
        {"config": config, "summary": summary, "results": results},
        run_dir / "report_data.json",
    )
    return report_path
