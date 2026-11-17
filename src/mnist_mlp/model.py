"""Многослойный перцептрон с настраиваемой шириной и активацией."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import nn

ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
}


class MLP(nn.Module):
    """Полносвязная сеть: 784 -> hidden... -> 10 логитов."""

    def __init__(
        self,
        hidden: Sequence[int] = (512, 256),
        activation: str = "relu",
        dropout: float = 0.2,
        batchnorm: bool = True,
        input_dim: int = 784,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(
                f"Неизвестная активация {activation!r}; доступны: {sorted(ACTIVATIONS)}"
            )
        if any(width <= 0 for width in hidden):
            raise ValueError(f"Ширина слоёв должна быть > 0, получено {hidden}")

        self.activation_name = activation
        self.input_dim = input_dim
        self.hidden = tuple(int(width) for width in hidden)

        layers: list[nn.Module] = []
        previous = input_dim
        for width in self.hidden:
            layers.append(nn.Linear(previous, width))
            if batchnorm:
                layers.append(nn.BatchNorm1d(width))
            layers.append(ACTIVATIONS[activation]())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous = width
        layers.append(nn.Linear(previous, num_classes))

        self.features = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if not isinstance(module, nn.Linear):
                continue
            if self.activation_name == "relu":
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_in", nonlinearity="relu"
                )
            elif self.activation_name == "tanh":
                nn.init.xavier_normal_(module.weight)
            else:
                nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)
        return self.features(x)

    def activation_modules(self) -> Iterable[nn.Module]:
        return (m for m in self.modules() if isinstance(m, tuple(ACTIVATIONS.values())))


def build_model(model_cfg: dict) -> MLP:
    return MLP(
        hidden=model_cfg.get("hidden", (512, 256)),
        activation=model_cfg.get("activation", "relu"),
        dropout=float(model_cfg.get("dropout", 0.2)),
        batchnorm=bool(model_cfg.get("batchnorm", True)),
    )
