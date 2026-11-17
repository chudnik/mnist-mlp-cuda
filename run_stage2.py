#!/usr/bin/env python3
"""Этап 2: обучение на 20 000 случайных объектах из train-пула.

E2 — tanh, E3 — ReLU. Каждый эксперимент повторяется на трёх seed'ах,
поэтому каждый seed получает своё случайное подмножество 20 000 объектов.

Запуск:
    .venv/bin/python run_stage2.py

Результаты: runs/stage2_<timestamp>/stage2.md, stage2.csv,
stage2_accuracy.png, stage2_data.json и подкаталоги прогонов.
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mnist_mlp.analysis import evaluate_split
from mnist_mlp.train import train
from mnist_mlp.utils import (
    load_config,
    load_json,
    project_path,
    save_json,
    timestamp,
)

SEEDS = (42, 43, 44)
EXPERIMENTS = (
    {
        "id": "E2",
        "name": "e2_tanh_20k",
        "label": "20k + tanh",
        "activation": "tanh",
        "config": "configs/stage2_tanh.yaml",
    },
    {
        "id": "E3",
        "name": "e3_relu_20k",
        "label": "20k + ReLU",
        "activation": "relu",
        "config": "configs/stage2_relu.yaml",
    },
)


def run_experiment(
    experiment: dict[str, str],
    seed: int,
    root: Path,
) -> dict[str, Any]:
    config = load_config(experiment["config"])
    config["seed"] = int(seed)
    config["output"]["dir"] = str(root / "runs")

    run_dir = train(
        config,
        name=f"{experiment['name']}_s{seed}",
        verbose=False,
    )
    summary = load_json(run_dir / "summary.json")
    results = evaluate_split(
        checkpoint_path=run_dir / "best.pt",
        split="test",
        out_dir=run_dir / "eval_test",
    )
    return {
        "experiment": experiment["id"],
        "label": experiment["label"],
        "activation": experiment["activation"],
        "seed": int(seed),
        "train_size": int(summary["data"]["train_size"]),
        "subset_hash": summary["data"]["train_subset_hash"],
        "val_acc": float(summary["best_val_acc"]),
        "test_acc": float(results["accuracy"]),
        "test_f1": float(results["macro_f1"]),
        "params": int(summary["params"]),
        "epochs": int(summary["epochs_ran"]),
        "seconds": float(summary["seconds"]),
        "run_dir": str(run_dir.relative_to(project_path("."))),
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for experiment in EXPERIMENTS:
        rows = [
            row for row in records if row["experiment"] == experiment["id"]
        ]
        if not rows:
            continue
        test = [row["test_acc"] for row in rows]
        val = [row["val_acc"] for row in rows]
        f1 = [row["test_f1"] for row in rows]
        stats[experiment["id"]] = {
            "label": experiment["label"],
            "train_size": rows[0]["train_size"],
            "n": len(rows),
            "test_mean": statistics.mean(test),
            "test_std": statistics.stdev(test) if len(test) > 1 else 0.0,
            "test_min": min(test),
            "val_mean": statistics.mean(val),
            "val_std": statistics.stdev(val) if len(val) > 1 else 0.0,
            "f1_mean": statistics.mean(f1),
            "params": rows[0]["params"],
            "epochs": statistics.mean(row["epochs"] for row in rows),
            "seconds": statistics.mean(row["seconds"] for row in rows),
        }
    return stats


def reference_54k(prefix: str) -> dict[str, Any] | None:
    """Берёт последний отчёт модели на полном train-пуле (54k)."""
    paths = sorted(
        project_path("runs").glob(f"{prefix}_*/report_data.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not paths:
        return None
    payload = load_json(paths[-1])
    return {
        "run": paths[-1].parent.name,
        "val_acc": float(payload["summary"]["best_val_acc"]),
        "test_acc": float(payload["results"]["test"]["accuracy"]),
        "test_f1": float(payload["results"]["test"]["macro_f1"]),
        "params": int(payload["summary"]["params"]),
    }


def write_outputs(
    root: Path,
    records: list[dict[str, Any]],
    stats: dict[str, dict[str, Any]],
    references: dict[str, dict[str, Any] | None],
) -> None:
    fields = [
        "experiment",
        "label",
        "activation",
        "seed",
        "train_size",
        "subset_hash",
        "val_acc",
        "test_acc",
        "test_f1",
        "params",
        "epochs",
        "seconds",
        "run_dir",
    ]
    with (root / "stage2.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({key: row[key] for key in fields})

    lines = [
        "# Этап 2: обучение на 20 000 случайных объектах",
        "",
        f"Seed'ы: {list(SEEDS)}. На каждом seed'е — своё случайное "
        "подмножество 20 000 объектов из train-пула 54 000.",
        "Валидация и test те же, что на этапе 1: 6 000 и 10 000.",
        "",
        "## Результаты",
        "",
        "| Эксперимент | Активация | Train | Test acc (mean ± std) | "
        "Test min | Val acc | Macro F1 | Параметров |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for experiment in EXPERIMENTS:
        values = stats.get(experiment["id"])
        if values is None:
            continue
        lines.append(
            f"| {experiment['id']} | {experiment['activation']} | "
            f"{values['train_size']:,} | "
            f"{values['test_mean'] * 100:.2f} ± "
            f"{values['test_std'] * 100:.2f} % | "
            f"{values['test_min'] * 100:.2f} % | "
            f"{values['val_mean'] * 100:.2f} % | "
            f"{values['f1_mean'] * 100:.2f} % | "
            f"{values['params']:,} |"
        )

    lines += ["", "## По seed'ам", ""]
    lines += [
        "| Эксперимент | Seed | Train | Val acc | Test acc | Test F1 |",
        "|---|---|---|---|---|---|",
    ]
    for row in records:
        lines.append(
            f"| {row['experiment']} | {row['seed']} | {row['train_size']:,} | "
            f"{row['val_acc'] * 100:.2f} % | {row['test_acc'] * 100:.2f} % | "
            f"{row['test_f1'] * 100:.2f} % |"
        )

    relu_54 = references.get("relu")
    tanh_54 = references.get("tanh")
    lines += ["", "## Сравнение с этапом 1 (54 000)", ""]
    lines += [
        "| Условие | Train | Активация | Test acc | Источник |",
        "|---|---|---|---|---|",
    ]
    if relu_54:
        lines.append(
            f"| E1 | 54 000 | relu | {relu_54['test_acc'] * 100:.2f} % | "
            f"`runs/{relu_54['run']}` |"
        )
    if tanh_54:
        lines.append(
            f"| E4 | 54 000 | tanh | {tanh_54['test_acc'] * 100:.2f} % | "
            f"`runs/{tanh_54['run']}` |"
        )
    for experiment in EXPERIMENTS:
        values = stats.get(experiment["id"])
        if values is None:
            continue
        lines.append(
            f"| {experiment['id']} | {values['train_size']:,} | "
            f"{experiment['activation']} | "
            f"{values['test_mean'] * 100:.2f} ± "
            f"{values['test_std'] * 100:.2f} % | этот запуск |"
        )

    lines += ["", "## Выводы", ""]
    if "E2" in stats and "E3" in stats:
        delta_activation = (
            stats["E2"]["test_mean"] - stats["E3"]["test_mean"]
        ) * 100
        lines.append(
            f"- При 20k tanh минус ReLU: {delta_activation:+.2f} п.п. "
            "Это и есть корректная оценка эффекта активации при малой выборке."
        )
    if "E3" in stats and relu_54:
        delta_size = (stats["E3"]["test_mean"] - relu_54["test_acc"]) * 100
        lines.append(
            f"- При ReLU 20k минус 54k: {delta_size:+.2f} п.п. "
            "Оценка вклада объёма обучающей выборки."
        )
    if "E2" in stats and tanh_54:
        delta_size_tanh = (stats["E2"]["test_mean"] - tanh_54["test_acc"]) * 100
        lines.append(
            f"- При tanh 20k минус 54k: {delta_size_tanh:+.2f} п.п. "
            "Оценка вклада объёма выборки для tanh."
        )
    lines += [
        "",
        "Разброс по seed'ам включает как различие инициализации, так и различие "
        "самих подмножеств 20 000 объектов.",
        "",
        "## Артефакты",
        "",
        "- `stage2.csv` — все прогоны",
        "- `stage2_accuracy.png` — сравнение mean ± std",
        "- `stage2_data.json` — агрегаты и пути",
        "- `runs/` — чекпоинты, логи и eval_test каждого прогона",
        "",
    ]
    (root / "stage2.md").write_text("\n".join(lines), encoding="utf-8")


def plot_results(
    root: Path,
    stats: dict[str, dict[str, Any]],
    references: dict[str, dict[str, Any] | None],
) -> None:
    labels = [stats[experiment["id"]]["label"] for experiment in EXPERIMENTS
              if experiment["id"] in stats]
    means = [stats[experiment["id"]]["test_mean"] * 100
             for experiment in EXPERIMENTS if experiment["id"] in stats]
    errors = [stats[experiment["id"]]["test_std"] * 100
              for experiment in EXPERIMENTS if experiment["id"] in stats]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    positions = range(len(labels))
    ax.bar(positions, means, yerr=errors, capsize=5, color="#2a6f97")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Test accuracy, %")
    ax.set_title("Этап 2: 20 000 объектов, mean ± std по 3 seed'ам")
    ax.grid(axis="y", alpha=0.3)
    for position, value in zip(positions, means):
        ax.text(position, value + 0.03, f"{value:.2f}", ha="center", fontsize=9)

    if references.get("relu"):
        ax.axhline(
            references["relu"]["test_acc"] * 100,
            color="#2a9d8f",
            linestyle="--",
            linewidth=1,
            label=f"E1 ReLU 54k: {references['relu']['test_acc'] * 100:.2f} %",
        )
    if references.get("tanh"):
        ax.axhline(
            references["tanh"]["test_acc"] * 100,
            color="#e07a5f",
            linestyle="--",
            linewidth=1,
            label=f"E4 tanh 54k: {references['tanh']['test_acc'] * 100:.2f} %",
        )
    ax.set_ylim(min(means) - 1.0, max(means) + 1.0)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "stage2_accuracy.png", dpi=140)
    plt.close(fig)


def main() -> None:
    root = project_path("runs") / f"stage2_{timestamp()}"
    root.mkdir(parents=True, exist_ok=True)

    print(
        f"Этап 2: 20 000 объектов, {len(EXPERIMENTS)} эксперимента, "
        f"seed'ы {list(SEEDS)}"
    )
    records: list[dict[str, Any]] = []
    for experiment in EXPERIMENTS:
        for seed in SEEDS:
            record = run_experiment(experiment, seed, root)
            records.append(record)
            print(
                f"{record['experiment']} {record['activation']:<4} "
                f"seed={seed} | train {record['train_size']:,} | "
                f"val {record['val_acc'] * 100:6.2f} % | "
                f"test {record['test_acc'] * 100:6.2f} % | "
                f"{record['epochs']} эпох, {record['seconds']:.1f} с"
            )

    stats = aggregate(records)
    references = {
        "relu": reference_54k("relu"),
        "tanh": reference_54k("tanh"),
    }
    write_outputs(root, records, stats, references)
    plot_results(root, stats, references)
    save_json(
        {
            "seeds": list(SEEDS),
            "experiments": [dict(experiment) for experiment in EXPERIMENTS],
            "records": records,
            "aggregates": stats,
            "references_54k": references,
        },
        root / "stage2_data.json",
    )

    print(f"\nСводка: {root / 'stage2.md'}")
    for experiment in EXPERIMENTS:
        values = stats.get(experiment["id"])
        if values:
            print(
                f"{experiment['id']} {experiment['activation']}: "
                f"test {values['test_mean'] * 100:.2f} ± "
                f"{values['test_std'] * 100:.2f} %"
            )


if __name__ == "__main__":
    main()
