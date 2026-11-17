#!/usr/bin/env python3
"""Подбор конфигурации E1 (54 000, ReLU) по validation.

Схема:
1. Скрининг всех кандидатов из configs/e1_search.yaml на одном seed.
2. Подтверждение top-k на нескольких seed'ах.
3. Победитель выбирается по средней val accuracy и один раз оценивается на test.
4. Сравнение сохраняется в comparison.md/.csv и val_accuracy.png.

Запуск:
    .venv/bin/python run_e1_selection.py
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

from mnist_mlp.analysis import evaluate_split
from mnist_mlp.report import build_report
from mnist_mlp.train import train
from mnist_mlp.utils import (
    load_config,
    load_json,
    project_path,
    save_json,
    timestamp,
)

BASE_CONFIG = "configs/base.yaml"
SEARCH_CONFIG = "configs/e1_search.yaml"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def run_candidate(
        base_config: dict[str, Any],
        candidate: dict[str, Any],
        candidate_config: dict[str, Any],
        seed: int,
        root: Path,
        phase: str,
) -> dict[str, Any]:
    config = deep_merge(base_config, candidate_config)
    config["seed"] = int(seed)
    config["output"]["dir"] = str(root / "runs")

    run_dir = train(
        config,
        name=f"{candidate['name']}_s{seed}",
        verbose=False,
    )
    summary = load_json(run_dir / "summary.json")
    return {
        "candidate": candidate["name"],
        "phase": phase,
        "seed": int(seed),
        "val_acc": float(summary["best_val_acc"]),
        "val_loss": float(summary["best_val_loss"]),
        "params": int(summary["params"]),
        "epochs": int(summary["epochs_ran"]),
        "seconds": float(summary["seconds"]),
        "hidden": config["model"]["hidden"],
        "dropout": float(config["model"]["dropout"]),
        "lr": float(config["training"]["lr"]),
        "weight_decay": float(config["training"]["weight_decay"]),
        "run_dir": str(run_dir.relative_to(project_path("."))),
    }


def aggregate(
        records: list[dict[str, Any]], names: list[str]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in names:
        rows = [row for row in records if row["candidate"] == name]
        if not rows:
            continue
        accuracies = [row["val_acc"] for row in rows]
        losses = [row["val_loss"] for row in rows]
        result[name] = {
            "n": len(rows),
            "mean_acc": statistics.mean(accuracies),
            "std_acc": statistics.stdev(accuracies) if len(rows) > 1 else 0.0,
            "mean_loss": statistics.mean(losses),
            "std_loss": statistics.stdev(losses) if len(rows) > 1 else 0.0,
            "mean_seconds": statistics.mean(row["seconds"] for row in rows),
            "params": rows[0]["params"],
        }
    return result


def write_comparison(
        root: Path,
        records: list[dict[str, Any]],
        stats: dict[str, dict[str, Any]],
        winner: str,
        screen_seed: int,
        top_k: int,
        results: dict[str, dict[str, Any]],
) -> None:
    fields = [
        "candidate",
        "phase",
        "seed",
        "val_acc",
        "val_loss",
        "params",
        "epochs",
        "seconds",
        "hidden",
        "dropout",
        "lr",
        "weight_decay",
        "run_dir",
    ]
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({key: row[key] for key in fields})

    lines = [
        "# Подбор конфигурации E1",
        "",
        f"Скрининг на seed={screen_seed}, подтверждение top-{top_k} на "
        f"{sorted({row['seed'] for row in records})}.",
        "Победитель выбран по средней val accuracy (при равенстве — по val loss). "
        "Test использован только для победителя.",
        "",
        f"**Победитель:** `{winner}`",
        "",
        "## Скрининг (все кандидаты)",
        "",
        "| Конфигурация | hidden | dropout | lr | wd | val acc | val loss | "
        "эпох | параметров |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    screening = sorted(
        [row for row in records if row["phase"] == "screen"],
        key=lambda row: (-row["val_acc"], row["val_loss"]),
    )
    for row in screening:
        hidden = "-".join(str(width) for width in row["hidden"])
        lines.append(
            f"| {row['candidate']} | {hidden} | {row['dropout']} | "
            f"{row['lr']:g} | {row['weight_decay']:g} | "
            f"{row['val_acc'] * 100:.2f} % | {row['val_loss']:.4f} | "
            f"{row['epochs']} | {row['params']:,} |"
        )

    lines += [
        "",
        f"## Подтверждение top-{top_k}",
        "",
        "| Конфигурация | seed'ов | val acc (mean ± std) | val loss | "
        "среднее время |",
        "|---|---|---|---|---|",
    ]
    for name, values in sorted(
            stats.items(), key=lambda item: (-item[1]["mean_acc"], item[1]["mean_loss"])
    ):
        marker = " **← победитель**" if name == winner else ""
        lines.append(
            f"| {name}{marker} | {values['n']} | "
            f"{values['mean_acc'] * 100:.2f} ± {values['std_acc'] * 100:.2f} % | "
            f"{values['mean_loss']:.4f} | {values['mean_seconds']:.1f} с |"
        )

    lines += [
        "",
        "## Итог победителя",
        "",
        "| Выборка | Accuracy | Macro F1 |",
        "|---|---|---|",
    ]
    for split in ("train", "val", "test"):
        lines.append(
            f"| {split} | {results[split]['accuracy'] * 100:.2f} % | "
            f"{results[split]['macro_f1'] * 100:.2f} % |"
        )

    spread = (
                     max(values["mean_acc"] for values in stats.values())
                     - min(values["mean_acc"] for values in stats.values())
             ) * 100
    lines += [
        "",
        "## Замечание о выборе",
        "",
        f"Разброс средних val accuracy между подтверждёнными конфигурациями — "
        f"{spread:.2f} п.п. Это сопоставимо со стандартной ошибкой оценки "
        f"(≈0.17 п.п. на test и ≈0.22 п.п. на val), поэтому выбор по validation "
        f"не гарантирует превосходство на test: возможен winner's curse. "
        f"Итоговые test-метрики приведены только для победителя, как и "
        f"предусмотрено протоколом.",
        "",
        "## Артефакты",
        "",
        "- `comparison.csv` — все прогоны",
        "- `val_accuracy.png` — сравнение подтверждённых конфигураций",
        "- `search_data.json` — агрегаты и пути к запускам",
        "- `runs/<кандидат>_s<seed>/` — чекпоинты, логи каждого прогона",
        f"- `runs/{winner}_s*/report.md` — полный отчёт победителя",
        "",
    ]
    (root / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


def plot_confirmation(
        root: Path, stats: dict[str, dict[str, Any]], winner: str
) -> None:
    names = sorted(stats, key=lambda name: stats[name]["mean_acc"])
    means = [stats[name]["mean_acc"] * 100 for name in names]
    errors = [stats[name]["std_acc"] * 100 for name in names]
    colors = ["#2a9d8f" if name == winner else "#8ab4c2" for name in names]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    positions = range(len(names))
    ax.bar(positions, means, yerr=errors, capsize=5, color=colors)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Validation accuracy, %")
    ax.set_title("E1: подтверждение лучших конфигураций")
    ax.grid(axis="y", alpha=0.3)
    for position, value in zip(positions, means):
        ax.text(position, value + 0.02, f"{value:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "val_accuracy.png", dpi=140)
    plt.close(fig)


def main() -> None:
    base_config = load_config(BASE_CONFIG)
    search = load_config(SEARCH_CONFIG)
    base_config = deep_merge(base_config, search.get("base_overrides", {}))

    candidates = search["candidates"]
    by_name = {candidate["name"]: candidate for candidate in candidates}
    screen_seed = int(search.get("screen_seed", 42))
    confirm_seeds = [int(seed) for seed in search.get("confirm_seeds", [42])]
    top_k = int(search.get("top_k", 3))

    root = project_path("runs") / f"e1_search_{timestamp()}"
    root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    print(f"Скрининг {len(candidates)} конфигураций (seed={screen_seed})")
    for index, candidate in enumerate(candidates, start=1):
        record = run_candidate(
            base_config,
            candidate,
            candidate.get("overrides", {}),
            screen_seed,
            root,
            phase="screen",
        )
        records.append(record)
        print(
            f"[{index:2d}/{len(candidates)}] {record['candidate']:<14} "
            f"val acc {record['val_acc'] * 100:6.2f} % "
            f"loss {record['val_loss']:.4f} "
            f"({record['seconds']:.1f} с, {record['epochs']} эпох)"
        )

    ranked = sorted(records, key=lambda row: (-row["val_acc"], row["val_loss"]))
    top_names = [row["candidate"] for row in ranked[:top_k]]

    print(f"\nПодтверждение top-{top_k}: {', '.join(top_names)}")
    for name in top_names:
        for seed in confirm_seeds:
            if any(
                    row["candidate"] == name and row["seed"] == seed
                    for row in records
            ):
                continue
            record = run_candidate(
                base_config,
                by_name[name],
                by_name[name].get("overrides", {}),
                seed,
                root,
                phase="confirm",
            )
            records.append(record)
            print(
                f"  {name:<14} seed={seed} "
                f"val acc {record['val_acc'] * 100:6.2f} % "
                f"loss {record['val_loss']:.4f}"
            )

    stats = aggregate(records, top_names)
    winner = max(
        stats,
        key=lambda name: (stats[name]["mean_acc"], -stats[name]["mean_loss"]),
    )
    print(
        f"\nПобедитель: {winner} — "
        f"val acc {stats[winner]['mean_acc'] * 100:.2f} ± "
        f"{stats[winner]['std_acc'] * 100:.2f} %"
    )

    winner_rows = [row for row in records if row["candidate"] == winner]
    winner_row = max(winner_rows, key=lambda row: row["val_acc"])
    winner_dir = project_path(winner_row["run_dir"])
    winner_config = load_config(winner_dir / "config.yaml")

    print("Финальная оценка победителя на train/val/test...")
    results = {
        split: evaluate_split(
            checkpoint_path=winner_dir / "best.pt",
            split=split,
            out_dir=winner_dir / f"eval_{split}",
        )
        for split in ("train", "val", "test")
    }
    report_path = build_report(winner_dir, results, winner_config)

    write_comparison(root, records, stats, winner, screen_seed, top_k, results)
    plot_confirmation(root, stats, winner)
    save_json(
        {
            "search_config": search,
            "records": records,
            "aggregates": stats,
            "winner": winner,
            "winner_run": winner_row["run_dir"],
            "winner_report": str(report_path.relative_to(project_path("."))),
            "winner_results": results,
        },
        root / "search_data.json",
    )

    print(f"\nСравнение: {root / 'comparison.md'}")
    print(f"Отчёт победителя: {report_path}")
    print(f"Test accuracy: {results['test']['accuracy'] * 100:.2f} %")


if __name__ == "__main__":
    main()
