#!/usr/bin/env python3
"""Обучение одной конфигурации и сбор данных для отчёта.

Запуск:
    .venv/bin/python run_training.py

Скрипт читает configs/relu.yaml, обучает MLP, оценивает лучший чекпоинт на
train/val/test и складывает все материалы для отчёта в runs/<name>_<time>/.
"""

from __future__ import annotations

from pathlib import Path

from mnist_mlp.analysis import evaluate_split
from mnist_mlp.report import build_report
from mnist_mlp.train import train
from mnist_mlp.utils import load_config

CONFIG_PATH = "configs/relu.yaml"
RUN_NAME = "relu"


def run(config_path: str = CONFIG_PATH, run_name: str = RUN_NAME) -> Path:
    config = load_config(config_path)
    run_dir = train(config, name=run_name)

    results = {}
    for split in ("train", "val", "test"):
        results[split] = evaluate_split(
            checkpoint_path=run_dir / "best.pt",
            split=split,
            out_dir=run_dir / f"eval_{split}",
        )

    report_path = build_report(run_dir, results, config)
    print(f"\nДанные для отчёта: {run_dir}")
    print(f"Markdown-отчёт:    {report_path}")
    return run_dir


def main() -> None:
    run()


if __name__ == "__main__":
    main()
