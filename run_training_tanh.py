#!/usr/bin/env python3
"""Обучение модели с активацией tanh.

Запуск:
    .venv/bin/python run_training_tanh.py

Скрипт читает configs/tanh.yaml и повторяет пайплайн run_training.py:
обучение, оценка на train/val/test и полный отчёт в runs/tanh_<time>/.
"""

from __future__ import annotations

from run_training import run

CONFIG_PATH = "configs/tanh.yaml"
RUN_NAME = "tanh"


def main() -> None:
    run(CONFIG_PATH, RUN_NAME)


if __name__ == "__main__":
    main()
