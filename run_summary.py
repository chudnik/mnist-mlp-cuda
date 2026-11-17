#!/usr/bin/env python3
"""Сводка по экспериментам E1–E5: таблицы, сходимость и графики.

Собирает уже выполненные прогоны:
- E1 — последний runs/relu_* (54k, ReLU, 46);
- E4 — последний runs/tanh_* (54k, tanh, 46);
- E2/E3 — последний runs/stage2_* (20k, tanh/ReLU, 3 seed'а);
- E5 — последний runs/min_neurons_* (минимум ширины, 5 seed'ов).

Запуск:
    .venv/bin/python run_summary.py

Результаты: runs/summary_<timestamp>/summary.md, summary.csv, runs.csv,
convergence.csv, learning_curves.png, summary_accuracy.png, summary_data.json.
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mnist_mlp.analysis import accuracy_on_split
from mnist_mlp.utils import load_json, project_path, save_json, timestamp

EXPERIMENT_ORDER = ("E1", "E2", "E3", "E4", "E5")
COLORS = {
    "E1": "#2a6f97",
    "E2": "#e07a5f",
    "E3": "#2a9d8f",
    "E4": "#8e7dbe",
    "E5": "#6c757d",
}


def _latest(pattern: str, filename: str) -> Path:
    paths = sorted(
        project_path("runs").glob(f"{pattern}_*/{filename}"),
        key=lambda path: path.stat().st_mtime,
    )
    if not paths:
        raise FileNotFoundError(f"Не найден runs/{pattern}_*/{filename}")
    return paths[-1]


def _load_metrics(run_dir: Path) -> list[dict[str, float]]:
    with (run_dir / "metrics.csv").open(encoding="utf-8") as fh:
        return [
            {
                "epoch": int(row["epoch"]),
                "train_loss": float(row["train_loss"]),
                "train_acc": float(row["train_acc"]),
                "val_loss": float(row["val_loss"]),
                "val_acc": float(row["val_acc"]),
                "seconds": float(row["seconds"]),
            }
            for row in csv.DictReader(fh)
        ]


def _first_epoch(
    rows: list[dict[str, float]], key: str, threshold: float
) -> int | None:
    for row in rows:
        if row[key] >= threshold:
            return row["epoch"]
    return None


def _train_accuracy(run_dir: Path) -> float:
    """Accuracy лучшего чекпоинта на train в eval-режиме."""
    return float(accuracy_on_split(run_dir / "best.pt", "train"))


def _report_records(experiment: str, pattern: str) -> list[dict[str, Any]]:
    run_dir = _latest(pattern, "report_data.json").parent
    payload = load_json(run_dir / "report_data.json")
    config = payload["config"]
    summary = payload["summary"]
    test = payload["results"]["test"]
    return [
        {
            "experiment": experiment,
            "train_size": int(summary["data"]["train_size"]),
            "activation": config["model"]["activation"],
            "neurons": int(config["model"]["hidden_neurons"]),
            "seed": int(config.get("seed", 42)),
            "test_acc": float(test["accuracy"]),
            "val_acc": float(summary["best_val_acc"]),
            "f1": float(test["macro_f1"]),
            "train_acc": _train_accuracy(run_dir),
            "epochs": int(summary["epochs_ran"]),
            "best_epoch": int(summary["best_epoch"]),
            "seconds": float(summary["seconds"]),
            "params": int(summary["params"]),
            "run_dir": str(run_dir.relative_to(project_path("."))),
        }
    ]


def _stage2_records() -> list[dict[str, Any]]:
    path = _latest("stage2", "stage2_data.json")
    payload = load_json(path)
    records: list[dict[str, Any]] = []
    for row in payload["records"]:
        run_dir = project_path(row["run_dir"])
        summary = load_json(run_dir / "summary.json")
        records.append(
            {
                "experiment": row["experiment"],
                "train_size": int(row["train_size"]),
                "activation": row["activation"],
                "neurons": 46,
                "seed": int(row["seed"]),
                "test_acc": float(row["test_acc"]),
                "val_acc": float(row["val_acc"]),
                "f1": float(row["test_f1"]),
                "train_acc": _train_accuracy(run_dir),
                "epochs": int(row["epochs"]),
                "best_epoch": int(summary["best_epoch"]),
                "seconds": float(row["seconds"]),
                "params": int(row["params"]),
                "run_dir": row["run_dir"],
            }
        )
    return records


def _e5_records() -> list[dict[str, Any]]:
    path = _latest("min_neurons", "search_data.json")
    payload = load_json(path)
    winner = int(payload["winner_neurons"])
    final_run = payload["final_run"]
    final_f1 = float(payload["results"]["test"]["macro_f1"])
    records: list[dict[str, Any]] = []
    for row in payload["records"]:
        if int(row["neurons"]) != winner:
            continue
        run_dir = project_path(row["run_dir"])
        summary = load_json(run_dir / "summary.json")
        records.append(
            {
                "experiment": "E5",
                "train_size": int(summary["data"]["train_size"]),
                "activation": "relu",
                "neurons": winner,
                "seed": int(row["seed"]),
                "test_acc": float(row["test_acc"]),
                "val_acc": float(row["val_acc"]),
                "f1": final_f1 if row["run_dir"] == final_run else None,
                "train_acc": _train_accuracy(run_dir),
                "epochs": int(row["epochs"]),
                "best_epoch": int(summary["best_epoch"]),
                "seconds": float(row["seconds"]),
                "params": int(row["params"]),
                "run_dir": row["run_dir"],
            }
        )
    return records


def aggregate(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for experiment in EXPERIMENT_ORDER:
        rows = [row for row in records if row["experiment"] == experiment]
        if not rows:
            continue
        test = [row["test_acc"] for row in rows]
        val = [row["val_acc"] for row in rows]
        train = [row["train_acc"] for row in rows]
        f1 = [row["f1"] for row in rows if row["f1"] is not None]
        stats[experiment] = {
            "label": (
                f"{rows[0]['train_size']:,}".replace(",", " ")
                + f" + {rows[0]['activation']}"
            ),
            "train_size": rows[0]["train_size"],
            "activation": rows[0]["activation"],
            "neurons": rows[0]["neurons"],
            "n": len(rows),
            "test_mean": statistics.mean(test),
            "test_std": statistics.stdev(test) if len(test) > 1 else 0.0,
            "test_min": min(test),
            "val_mean": statistics.mean(val),
            "val_std": statistics.stdev(val) if len(val) > 1 else 0.0,
            "train_mean": statistics.mean(train),
            "f1_mean": statistics.mean(f1) if f1 else None,
            "f1_std": statistics.stdev(f1) if len(f1) > 1 else 0.0,
            "gap": statistics.mean(train) - statistics.mean(test),
            "epochs": statistics.mean(row["epochs"] for row in rows),
            "seconds": statistics.mean(row["seconds"] for row in rows),
            "params": rows[0]["params"],
            "records": rows,
        }
    return stats


def convergence(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for experiment in EXPERIMENT_ORDER:
        rows = [row for row in records if row["experiment"] == experiment]
        if not rows:
            continue
        epochs_95: list[int] = []
        epochs_98: list[int] = []
        for row in rows:
            metrics = _load_metrics(project_path(row["run_dir"]))
            first_95 = _first_epoch(metrics, "val_acc", 0.95)
            first_98 = _first_epoch(metrics, "val_acc", 0.98)
            if first_95 is not None:
                epochs_95.append(first_95)
            if first_98 is not None:
                epochs_98.append(first_98)
        result[experiment] = {
            "n": len(rows),
            "epoch_95": statistics.mean(epochs_95) if epochs_95 else None,
            "reached_95": len(epochs_95),
            "epoch_98": statistics.mean(epochs_98) if epochs_98 else None,
            "reached_98": len(epochs_98),
            "seconds_per_epoch": statistics.mean(
                row["seconds"] / max(row["epochs"], 1) for row in rows
            ),
        }
    return result


def _fmt_acc(mean: float, std: float, n: int) -> str:
    if n > 1:
        return f"{mean * 100:.2f} ± {std * 100:.2f} %"
    return f"{mean * 100:.2f} %"


def _fmt_epoch(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.1f}"


def write_tables(
    root: Path,
    stats: dict[str, dict[str, Any]],
    records: list[dict[str, Any]],
    conv: dict[str, dict[str, Any]],
) -> None:
    summary_fields = [
        "experiment",
        "train_size",
        "activation",
        "neurons",
        "seeds",
        "test_acc",
        "test_std",
        "test_min",
        "val_acc",
        "val_std",
        "macro_f1",
        "train_acc",
        "gap",
        "epochs",
        "seconds",
        "params",
    ]
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=summary_fields)
        writer.writeheader()
        for experiment in EXPERIMENT_ORDER:
            values = stats.get(experiment)
            if not values:
                continue
            writer.writerow(
                {
                    "experiment": experiment,
                    "train_size": values["train_size"],
                    "activation": values["activation"],
                    "neurons": values["neurons"],
                    "seeds": values["n"],
                    "test_acc": f"{values['test_mean']:.6f}",
                    "test_std": f"{values['test_std']:.6f}",
                    "test_min": f"{values['test_min']:.6f}",
                    "val_acc": f"{values['val_mean']:.6f}",
                    "val_std": f"{values['val_std']:.6f}",
                    "macro_f1": (
                        "" if values["f1_mean"] is None
                        else f"{values['f1_mean']:.6f}"
                    ),
                    "train_acc": f"{values['train_mean']:.6f}",
                    "gap": f"{values['gap']:.6f}",
                    "epochs": f"{values['epochs']:.1f}",
                    "seconds": f"{values['seconds']:.1f}",
                    "params": values["params"],
                }
            )

    run_fields = [
        "experiment",
        "train_size",
        "activation",
        "neurons",
        "seed",
        "test_acc",
        "val_acc",
        "macro_f1",
        "train_acc",
        "gap",
        "best_epoch",
        "epochs",
        "seconds",
        "params",
        "run_dir",
    ]
    with (root / "runs.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=run_fields)
        writer.writeheader()
        for row in sorted(records, key=lambda item: (
            EXPERIMENT_ORDER.index(item["experiment"]), item["seed"]
        )):
            writer.writerow(
                {
                    "experiment": row["experiment"],
                    "train_size": row["train_size"],
                    "activation": row["activation"],
                    "neurons": row["neurons"],
                    "seed": row["seed"],
                    "test_acc": f"{row['test_acc']:.6f}",
                    "val_acc": f"{row['val_acc']:.6f}",
                    "macro_f1": (
                        "" if row["f1"] is None else f"{row['f1']:.6f}"
                    ),
                    "train_acc": f"{row['train_acc']:.6f}",
                    "gap": f"{row['train_acc'] - row['test_acc']:.6f}",
                    "best_epoch": row["best_epoch"],
                    "epochs": row["epochs"],
                    "seconds": f"{row['seconds']:.2f}",
                    "params": row["params"],
                    "run_dir": row["run_dir"],
                }
            )

    conv_fields = [
        "experiment",
        "runs",
        "epoch_95",
        "reached_95",
        "epoch_98",
        "reached_98",
        "seconds_per_epoch",
    ]
    with (root / "convergence.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=conv_fields)
        writer.writeheader()
        for experiment in EXPERIMENT_ORDER:
            values = conv.get(experiment)
            if not values:
                continue
            writer.writerow(
                {
                    "experiment": experiment,
                    "runs": values["n"],
                    "epoch_95": "" if values["epoch_95"] is None
                    else f"{values['epoch_95']:.1f}",
                    "reached_95": values["reached_95"],
                    "epoch_98": "" if values["epoch_98"] is None
                    else f"{values['epoch_98']:.1f}",
                    "reached_98": values["reached_98"],
                    "seconds_per_epoch": f"{values['seconds_per_epoch']:.2f}",
                }
            )

    lines = [
        "# Сводка экспериментов E1–E5",
        "",
        "Все прогоны используют одни и те же validation 6 000 и test 10 000.",
        "E1 и E4 — 54 000 объектов, 1 seed; E2/E3 — 20 000, 3 seed'а; "
        "E5 — поиск ширины, 5 seed'ов на победившей ширине.",
        "",
        "## Сводная таблица",
        "",
        "| Эксперимент | Train | Активация | Нейронов | Seed'ов | "
        "Test acc | Val acc | Macro F1 | Train acc | Зазор | Эпох | "
        "Время, с | Параметров |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for experiment in EXPERIMENT_ORDER:
        values = stats.get(experiment)
        if not values:
            continue
        f1 = (
            "—" if values["f1_mean"] is None
            else _fmt_acc(values["f1_mean"], values["f1_std"], len(
                [row for row in values["records"] if row["f1"] is not None]
            ))
        )
        lines.append(
            f"| {experiment} | {values['train_size']:,} | "
            f"{values['activation']} | {values['neurons']} | {values['n']} | "
            f"{_fmt_acc(values['test_mean'], values['test_std'], values['n'])} | "
            f"{_fmt_acc(values['val_mean'], values['val_std'], values['n'])} | "
            f"{f1} | {values['train_mean'] * 100:.2f} % | "
            f"{values['gap'] * 100:.2f} п.п. | {values['epochs']:.1f} | "
            f"{values['seconds']:.1f} | {values['params']:,} |"
        )

    lines += [
        "",
        "Примечания:",
        "- E1/E4 — один прогон, поэтому без std.",
        "- E5 — среднее по 5 seed'ам на минимальной ширине; Macro F1 взят "
        "из финального отчёта выбранного прогона.",
        "- У E2/E3 каждый seed использует своё случайное подмножество 20 000.",
        "",
        "## Сходимость",
        "",
        "| Эксперимент | Прогонов | До val 95 % | Достигли 95 % | "
        "До val 98 % | Достигли 98 % | с/эпоха |",
        "|---|---|---|---|---|---|---|",
    ]
    for experiment in EXPERIMENT_ORDER:
        values = conv.get(experiment)
        if not values:
            continue
        lines.append(
            f"| {experiment} | {values['n']} | "
            f"{_fmt_epoch(values['epoch_95'])} | {values['reached_95']}/"
            f"{values['n']} | {_fmt_epoch(values['epoch_98'])} | "
            f"{values['reached_98']}/{values['n']} | "
            f"{values['seconds_per_epoch']:.2f} |"
        )

    lines += [
        "",
        "«До val 95/98 %» — средняя эпоха первого достижения порога по тем "
        "прогонам, где он достигнут; «Достигли» — сколько прогонов его достигли.",
        "",
        "## По прогонам",
        "",
        "| Эксперимент | Seed | Test acc | Val acc | Train acc | Зазор | "
        "Лучшая эпоха | Эпох |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in sorted(records, key=lambda item: (
        EXPERIMENT_ORDER.index(item["experiment"]), item["seed"]
    )):
        lines.append(
            f"| {row['experiment']} | {row['seed']} | "
            f"{row['test_acc'] * 100:.2f} % | {row['val_acc'] * 100:.2f} % | "
            f"{row['train_acc'] * 100:.2f} % | "
            f"{(row['train_acc'] - row['test_acc']) * 100:.2f} п.п. | "
            f"{row['best_epoch']} | {row['epochs']} |"
        )

    lines += [
        "",
        "## Артефакты",
        "",
        "- `summary.csv` — агрегаты по экспериментам",
        "- `runs.csv` — все прогоны",
        "- `convergence.csv` — эпохи достижения val 95/98 %",
        "- `learning_curves.png` — loss и accuracy по эпохам (seed 42)",
        "- `summary_accuracy.png` — test accuracy по экспериментам",
        "- `summary_data.json` — все числа и пути",
        "",
    ]
    (root / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def plot_learning_curves(root: Path, records: list[dict[str, Any]]) -> None:
    representatives: dict[str, dict[str, Any]] = {}
    for experiment in ("E1", "E2", "E3", "E4"):
        candidates = [
            row for row in records
            if row["experiment"] == experiment and row["seed"] == 42
        ]
        if candidates:
            representatives[experiment] = candidates[0]

    if not representatives:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for experiment, row in representatives.items():
        metrics = _load_metrics(project_path(row["run_dir"]))
        epochs = [item["epoch"] for item in metrics]
        color = COLORS[experiment]
        label = (
            f"{experiment}: {row['train_size'] // 1000}k + {row['activation']}"
        )
        axes[0].plot(
            epochs, [item["val_loss"] for item in metrics],
            color=color, linewidth=1.8, label=label,
        )
        axes[0].plot(
            epochs, [item["train_loss"] for item in metrics],
            color=color, linewidth=1.1, linestyle="--", alpha=0.5,
        )
        axes[1].plot(
            epochs, [item["val_acc"] * 100 for item in metrics],
            color=color, linewidth=1.8, label=label,
        )
        axes[1].plot(
            epochs, [item["train_acc"] * 100 for item in metrics],
            color=color, linewidth=1.1, linestyle="--", alpha=0.5,
        )

    axes[0].set_xlabel("Эпоха")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss: сплошная — val, пунктир — train")
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("Эпоха")
    axes[1].set_ylabel("Accuracy, %")
    axes[1].set_title("Accuracy: сплошная — val, пунктир — train")
    axes[1].axhline(95.0, color="gray", linestyle=":", linewidth=1)
    axes[1].axhline(98.0, color="gray", linestyle=":", linewidth=1)
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(root / "learning_curves.png", dpi=140)
    plt.close(fig)


def plot_summary(root: Path, stats: dict[str, dict[str, Any]]) -> None:
    experiments = [name for name in EXPERIMENT_ORDER if name in stats]
    means = [stats[name]["test_mean"] * 100 for name in experiments]
    errors = [stats[name]["test_std"] * 100 for name in experiments]
    val_means = [stats[name]["val_mean"] * 100 for name in experiments]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    positions = range(len(experiments))
    ax.bar(
        positions, means, yerr=errors, capsize=5,
        color=[COLORS[name] for name in experiments], alpha=0.85,
        label="test accuracy",
    )
    ax.scatter(
        positions, val_means, color="black", marker="x", s=55,
        zorder=3, label="val accuracy",
    )
    ax.axhline(97.0, color="black", linestyle="--", linewidth=1, label="порог 97 %")
    labels = [
        f"{name}\n{stats[name]['train_size'] // 1000}k "
        f"{stats[name]['activation']}"
        for name in experiments
    ]
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy, %")
    ax.set_title("E1–E5: test accuracy (mean ± std) и val accuracy")
    ax.grid(axis="y", alpha=0.3)
    for position, value in zip(positions, means):
        ax.text(
            position, value + 0.15, f"{value:.2f}",
            ha="center", fontsize=9,
        )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "summary_accuracy.png", dpi=140)
    plt.close(fig)


def main() -> None:
    records = (
        _report_records("E1", "relu")
        + _stage2_records()
        + _report_records("E4", "tanh")
        + _e5_records()
    )
    stats = aggregate(records)
    conv = convergence(records)

    root = project_path("runs") / f"summary_{timestamp()}"
    root.mkdir(parents=True, exist_ok=True)

    write_tables(root, stats, records, conv)
    plot_learning_curves(root, records)
    plot_summary(root, stats)
    save_json(
        {
            "aggregates": stats,
            "convergence": conv,
            "records": records,
        },
        root / "summary_data.json",
    )

    print(f"Сводка: {root / 'summary.md'}")
    for experiment in EXPERIMENT_ORDER:
        values = stats.get(experiment)
        if values:
            print(
                f"{experiment}: test "
                f"{values['test_mean'] * 100:.2f} ± "
                f"{values['test_std'] * 100:.2f} %"
            )
    print(f"Графики: {root / 'learning_curves.png'}, "
          f"{root / 'summary_accuracy.png'}")


if __name__ == "__main__":
    main()
