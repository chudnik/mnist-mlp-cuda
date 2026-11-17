"""Вспомогательные функции: конфиг, seed, устройство, вывод.

Здесь нет обучения — только инфраструктура, которую используют остальные
модули: пути проекта, чтение/запись YAML и JSON, фиксация seed, выбор
устройства (CPU/CUDA), форматирование описания GPU и счётчик параметров.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

# Корень репозитория: файл лежит в mnist_mlp/utils.py, поэтому поднимаемся
# на два уровня вверх. От него строятся все относительные пути (configs/, runs/).
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str | Path) -> Path:
    """Преобразует путь в абсолютный относительно корня проекта.

    Абсолютные пути и ``~`` возвращаются как есть. Нужно, чтобы скрипты
    можно было запускать из любого каталога, а не только из корня репозитория.
    """
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: str | Path = "configs/relu.yaml") -> dict[str, Any]:
    """Читает YAML-конфиг и проверяет, что верхний уровень — словарь.

    Конфиг содержит секции data, model, training и output; именно их
    используют train(), build_datasets() и build_model().
    """
    config_path = project_path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    if not isinstance(config, dict):
        raise ValueError(f"Конфиг {config_path} должен быть YAML-словарём")
    return config


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """Сохраняет конфиг рядом с артефактами запуска (config.yaml).

    Это нужно для воспроизводимости: по сохранённому конфигу можно точно
    понять, с какими гиперпараметрами получен чекпоинт.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False, allow_unicode=True)


def save_json(payload: Any, path: str | Path) -> None:
    """Сохраняет произвольный Python-объект в JSON с отступами.

    ensure_ascii=False оставляет русский текст читаемым, indent=2 делает
    файл удобным для просмотра преподавателем.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Any:
    """Читает JSON-файл (summary.json, metrics.json, search_data.json и т.п.)."""
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Фиксирует все источники случайности для воспроизводимых прогонов.

    random/numpy/torch отвечают за инициализацию весов, перемешивание данных
    и dropout. deterministic=True дополнительно включает детерминированные
    алгоритмы cuDNN (медленнее, но результат стабильнее).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(preference: str = "auto") -> torch.device:
    """Выбирает устройство вычислений: cpu, cuda или auto.

    auto использует CUDA, если она доступна, иначе CPU. Если CUDA запрошена
    явно, но недоступна, выбрасывается понятная ошибка вместо тихого
    перехода на CPU.
    """
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA запрошена явно, но torch.cuda.is_available() == False. "
                "Проверьте установку CUDA-сборки PyTorch."
            )
        return torch.device("cuda")
    if preference not in ("auto", ""):
        raise ValueError(f"Неизвестное устройство: {preference!r}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe_device(device: torch.device) -> str:
    """Возвращает человекочитаемое описание устройства для логов и отчёта.

    Для CUDA показывает индекс, имя GPU, compute capability, объём памяти
    и версии PyTorch/CUDA — это полезно фиксировать в отчёте.
    """
    if device.type != "cuda":
        return "CPU (CUDA недоступна)"
    index = device.index or 0
    name = torch.cuda.get_device_name(index)
    capability = torch.cuda.get_device_capability(index)
    total_gb = torch.cuda.get_device_properties(index).total_memory / 1024 ** 3
    return (
        f"CUDA:{index} {name} sm_{capability[0]}{capability[1]}, "
        f"{total_gb:.1f} ГБ, torch {torch.__version__}, CUDA {torch.version.cuda}"
    )


def timestamp() -> str:
    """Текущее время в формате YYYYMMDD-HHMMSS для имени каталога запуска."""
    return time.strftime("%Y%m%d-%H%M%S")


def count_parameters(model: torch.nn.Module) -> int:
    """Считает число обучаемых параметров модели.

    Используется в логах, summary.json и сводных таблицах, чтобы сравнивать
    модели по размеру, а не только по точности.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
