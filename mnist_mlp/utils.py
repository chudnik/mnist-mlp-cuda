"""Вспомогательные функции: конфиг, seed, устройство, вывод."""

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str | Path) -> Path:
    """Путь относительно корня проекта, если он не абсолютный."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: str | Path = "configs/base.yaml") -> dict[str, Any]:
    config_path = project_path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    if not isinstance(config, dict):
        raise ValueError(f"Конфиг {config_path} должен быть YAML-словарём")
    return config


def save_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False, allow_unicode=True)


def save_json(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(preference: str = "auto") -> torch.device:
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
    return time.strftime("%Y%m%d-%H%M%S")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
