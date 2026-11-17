#!/usr/bin/env python3
"""Поиск минимального числа нейронов одного скрытого слоя.

Алгоритм:
- начинаем с 32 нейронов;
- на каждой ширине обучаем модель на 5 seed'ах;
- считаем test accuracy каждой модели;
- увеличиваем ширину на step (1 нейрон), пока test accuracy **каждого** seed'а
  не достигнет 97 %;
- для найденной ширины собираем полный отчёт.

Запуск:
    .venv/bin/python run_min_neurons.py
"""

from __future__ import annotations

import copy
import csv
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mnist_mlp.analysis import accuracy_on_split, evaluate_split
from mnist_mlp.report import build_report
from mnist_mlp.train import train
from mnist_mlp.utils import (
    load_config,
    load_json,
    project_path,
    save_json,
    timestamp,
)

BASE_CONFIG = "configs/relu.yaml"
SEARCH_CONFIG = "configs/min_neurons.yaml"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def find_min_hidden_neurons(
    base_config: dict[str, Any],
    search: dict[str, Any],
    root: Path,
) -> tuple[int | None, list[dict[str, Any]]]:
    """Идёт от start_neurons вверх и возвращает первую прошедшую ширину."""
    start = int(search["start_neurons"])
    step = int(search["step"])
    max_neurons = int(search["max_neurons"])
    seeds = [int(seed) for seed in search["seeds"]]
    target = float(search["target_accuracy"])
    criterion = str(search.get("criterion", "mean"))

    records: list[dict[str, Any]] = []
    found: int | None = None

    for width in range(start, max_neurons + 1, step):
        width_records: list[dict[str, Any]] = []
        for seed in seeds:
            config = deep_merge(
                base_config,
                {
                    "model": {
                        "hidden_neurons": width,
                        "activation": "relu",
                        "dropout": 0.2,
                        "batchnorm": True,
                    },
                    "seed": seed,
                    "output": {"dir": str(root / "runs")},
                },
            )
            run_dir = train(
                config,
                name=f"n{width:03d}_s{seed}",
                verbose=False,
            )
            summary = load_json(run_dir / "summary.json")
            test_acc = accuracy_on_split(run_dir / "best.pt", "test")
            record = {
                "neurons": width,
                "seed": seed,
                "val_acc": float(summary["best_val_acc"]),
                "test_acc": test_acc,
                "params": int(summary["params"]),
                "epochs": int(summary["epochs_ran"]),
                "seconds": float(summary["seconds"]),
                "run_dir": str(run_dir.relative_to(project_path("."))),
            }
            records.append(record)
            width_records.append(record)

        test_values = [record["test_acc"] for record in width_records]
        val_values = [record["val_acc"] for record in width_records]
        mean_test = statistics.mean(test_values)
        std_test = statistics.stdev(test_values)
        mean_val = statistics.mean(val_values)
        print(
            f"нейронов {width:3d} | "
            f"val {mean_val * 100:6.2f} % | "
            f"test {mean_test * 100:6.2f} ± {std_test * 100:.2f} % | "
            f"min {min(test_values) * 100:6.2f} % | "
            f"параметров {width_records[0]['params']:,}"
        )

        passed = (
            all(value >= target for value in test_values)
            if criterion == "all"
            else mean_test >= target
        )
        if passed:
            found = width
            break

    return found, records


def write_results(
    root: Path,
    search: dict[str, Any],
    winner: int,
    records: list[dict[str, Any]],
    results: dict[str, dict[str, Any]],
) -> None:
    seeds = [int(seed) for seed in search["seeds"]]
    target = float(search["target_accuracy"])

    with (root / "search.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "neurons",
                "seed",
                "val_acc",
                "test_acc",
                "params",
                "epochs",
                "seconds",
                "run_dir",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    widths = sorted({record["neurons"] for record in records})
    lines = [
        "# Минимальное число нейронов скрытого слоя",
        "",
        f"Старт: {search['start_neurons']}, шаг: {search['step']}, "
        f"seed'ы: {seeds}, порог: {target * 100:.0f} % ({search['criterion']}).",
        "",
        f"**Результат: {winner} нейронов в одном скрытом слое.**",
        "",
        "## По ширине",
        "",
        "| Нейронов | Параметров | Val acc | Test acc | Test min |",
        "|---|---|---|---|---|",
    ]
    for width in widths:
        rows = [record for record in records if record["neurons"] == width]
        test = [record["test_acc"] for record in rows]
        val = [record["val_acc"] for record in rows]
        lines.append(
            f"| {width} | {rows[0]['params']:,} | "
            f"{statistics.mean(val) * 100:.2f} % | "
            f"{statistics.mean(test) * 100:.2f} ± "
            f"{statistics.stdev(test) * 100:.2f} % | "
            f"{min(test) * 100:.2f} % |"
        )

    lines += [
        "",
        f"## Модель-победитель: {winner} нейронов",
        "",
        "| Seed | Val acc | Test acc |",
        "|---|---|---|",
    ]
    for record in [row for row in records if row["neurons"] == winner]:
        lines.append(
            f"| {record['seed']} | {record['val_acc'] * 100:.2f} % | "
            f"{record['test_acc'] * 100:.2f} % |"
        )

    lines += [
        "",
        "## Итоговые метрики",
        "",
        "| Выборка | Accuracy | Macro F1 |",
        "|---|---|---|",
    ]
    for split in ("train", "val", "test"):
        lines.append(
            f"| {split} | {results[split]['accuracy'] * 100:.2f} % | "
            f"{results[split]['macro_f1'] * 100:.2f} % |"
        )

    smaller = [
        record
        for record in records
        if record["neurons"] < winner
    ]
    if smaller:
        lines += [
            "",
            "## Почему меньше не подошло",
            "",
            "| Нейронов | Test acc (mean) |",
            "|---|---|",
        ]
        for width in sorted({record["neurons"] for record in smaller}):
            values = [
                record["test_acc"]
                for record in smaller
                if record["neurons"] == width
            ]
            lines.append(
                f"| {width} | {statistics.mean(values) * 100:.2f} % |"
            )

    lines += [
        "",
        "## Артефакты",
        "",
        "- `search.csv` — все прогоны по seed'ам",
        "- `accuracy_vs_neurons.png` — кривая точности",
        "- `search_data.json` — агрегаты и пути",
        "- `runs/` — чекпоинты и логи каждого прогона",
        "",
    ]
    (root / "search.md").write_text("\n".join(lines), encoding="utf-8")


def plot_curve(root: Path, records: list[dict[str, Any]], winner: int) -> None:
    widths = sorted({record["neurons"] for record in records})
    means = []
    errors = []
    for width in widths:
        values = [
            record["test_acc"] for record in records if record["neurons"] == width
        ]
        means.append(statistics.mean(values) * 100)
        errors.append(statistics.stdev(values) * 100)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.errorbar(
        widths,
        means,
        yerr=errors,
        fmt="o-",
        capsize=4,
        color="#2a6f97",
        label="test accuracy, mean ± std (5 seed'ов)",
    )
    ax.axhline(97.0, color="black", linestyle="--", linewidth=1)
    index = widths.index(winner)
    ax.scatter(
        [widths[index]],
        [means[index]],
        s=160,
        facecolors="none",
        edgecolors="#2a9d8f",
        linewidths=2,
        label=f"минимум: {winner}",
    )
    ax.set_xlabel("Нейроны скрытого слоя")
    ax.set_ylabel("Test accuracy, %")
    ax.set_title("Минимальная ширина скрытого слоя")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "accuracy_vs_neurons.png", dpi=140)
    plt.close(fig)


def main() -> None:
    base_config = load_config(BASE_CONFIG)
    search = load_config(SEARCH_CONFIG)
    base_config = deep_merge(base_config, search.get("base_overrides", {}))

    root = project_path("runs") / f"min_neurons_{timestamp()}"
    root.mkdir(parents=True, exist_ok=True)

    print(
        f"Поиск от {search['start_neurons']} нейронов с шагом {search['step']}, "
        f"{len(search['seeds'])} seed'ов, порог {search['target_accuracy'] * 100:.0f} %"
    )
    winner, records = find_min_hidden_neurons(base_config, search, root)

    if winner is None:
        print("Порог не достигнут в заданном диапазоне.")
        save_json({"search": search, "records": records}, root / "search_data.json")
        return

    print(f"\nМинимальная ширина: {winner} нейронов")

    winner_records = [record for record in records if record["neurons"] == winner]
    median_test = statistics.median(
        record["test_acc"] for record in winner_records
    )
    final_record = min(
        winner_records,
        key=lambda record: abs(record["test_acc"] - median_test),
    )
    winner_dir = project_path(final_record["run_dir"])
    winner_config = load_config(winner_dir / "config.yaml")

    print(f"Финальная модель: seed={final_record['seed']}")
    results = {
        split: evaluate_split(
            checkpoint_path=winner_dir / "best.pt",
            split=split,
            out_dir=winner_dir / f"eval_{split}",
        )
        for split in ("train", "val", "test")
    }
    report_path = build_report(winner_dir, results, winner_config)

    write_results(root, search, winner, records, results)
    plot_curve(root, records, winner)
    save_json(
        {
            "search": search,
            "winner_neurons": winner,
            "final_seed": final_record["seed"],
            "final_run": final_record["run_dir"],
            "final_report": str(report_path.relative_to(project_path("."))),
            "records": records,
            "results": results,
        },
        root / "search_data.json",
    )

    print(f"\nСравнение: {root / 'search.md'}")
    print(f"Отчёт модели: {report_path}")
    print(f"Test accuracy: {results['test']['accuracy'] * 100:.2f} %")


if __name__ == "__main__":
    main()
