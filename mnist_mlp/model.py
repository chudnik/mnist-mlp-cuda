"""Однослойный скрытый MLP с настраиваемым числом нейронов.

Модель принципиально простая: вход 784 (28×28), один скрытый слой и
10 выходных логитов (по одному на цифру). Глубина фиксирована, меняется
только ширина скрытого слоя — это требование эксперимента E5.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

# Поддерживаемые активации. Словарь позволяет выбрать класс по строке из YAML.
ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
}


class MLP(nn.Module):
    """Полносвязная сеть 784 -> hidden_neurons -> 10.

    Порядок слоёв скрытого блока:
        Linear -> BatchNorm (опционально) -> активация -> Dropout (опционально)
    Затем идёт выходной Linear без активации: он выдаёт логиты, а мягкий
    максимум применяется внутри CrossEntropyLoss при обучении.
    """

    def __init__(
        self,
        hidden_neurons: int = 512,
        activation: str = "relu",
        dropout: float = 0.2,
        batchnorm: bool = True,
        input_dim: int = 784,
        num_classes: int = 10,
    ) -> None:
        """Создаёт слои модели по числу нейронов и настройкам регуляризации.

        Аргументы:
            hidden_neurons: ширина единственного скрытого слоя.
            activation: имя активации из ACTIVATIONS.
            dropout: вероятность отключения нейрона (0 отключает Dropout).
            batchnorm: добавлять ли BatchNorm1d после скрытого Linear.
            input_dim: размер входа (784 для MNIST 28×28).
            num_classes: число классов (10 цифр).
        """
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(
                f"Неизвестная активация {activation!r}; доступны: {sorted(ACTIVATIONS)}"
            )
        if hidden_neurons <= 0:
            raise ValueError(
                f"Число нейронов скрытого слоя должно быть > 0, "
                f"получено {hidden_neurons}"
            )

        # Запоминаем настройки: они нужны для инициализации и для отчёта.
        self.activation_name = activation
        self.input_dim = input_dim
        self.hidden_neurons = int(hidden_neurons)

        layers: list[nn.Module] = [
            nn.Linear(input_dim, self.hidden_neurons),
        ]
        if batchnorm:
            layers.append(nn.BatchNorm1d(self.hidden_neurons))
        layers.append(ACTIVATIONS[activation]())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(self.hidden_neurons, num_classes))

        # Sequential хранит блоки в том порядке, в котором они применяются.
        self.features = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        """Инициализирует веса Linear-слоёв.

        Для ReLU используется Kaiming/He, для tanh и gelu — Xavier/Glorot.
        Смещения обнуляются: это стандартная практика, BatchNorm всё равно
        сдвигает активации.
        """
        for module in self.modules():
            if not isinstance(module, nn.Linear):
                continue
            if self.activation_name == "relu":
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_in", nonlinearity="relu"
                )
            else:
                nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход: изображение (N,1,28,28) или (N,784) -> логиты (N,10)."""
        # flatten(1) превращает каждый тензор изображения в вектор длины 784.
        x = x.flatten(1)
        return self.features(x)

    def activation_modules(self) -> Iterable[nn.Module]:
        """Перебирает модули активации (используется для мониторинга tanh)."""
        return (
            module
            for module in self.modules()
            if isinstance(module, tuple(ACTIVATIONS.values()))
        )


def build_model(model_cfg: dict) -> MLP:
    """Собирает MLP по секции model из YAML-конфига.

    Ожидает обязательный ключ hidden_neurons (число нейронов скрытого слоя)
    и опциональные activation, dropout, batchnorm. Списки слоёв больше не
    поддерживаются: по условию задачи скрытый слой всегда один.
    """
    if "hidden_neurons" not in model_cfg:
        raise ValueError("В конфиге модели нет hidden_neurons (число нейронов)")

    return MLP(
        hidden_neurons=int(model_cfg["hidden_neurons"]),
        activation=model_cfg.get("activation", "relu"),
        dropout=float(model_cfg.get("dropout", 0.2)),
        batchnorm=bool(model_cfg.get("batchnorm", True)),
    )
