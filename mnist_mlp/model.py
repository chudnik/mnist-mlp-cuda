"""Однослойный скрытый MLP с настраиваемым числом нейронов."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
}


class MLP(nn.Module):
    """784 -> hidden_neurons -> 10 логитов.

    Скрытый слой всегда один; его шириной управляет hidden_neurons.
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
            else:
                nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)
        return self.features(x)

    def activation_modules(self) -> Iterable[nn.Module]:
        return (
            module
            for module in self.modules()
            if isinstance(module, tuple(ACTIVATIONS.values()))
        )


def build_model(model_cfg: dict) -> MLP:
    if "hidden_neurons" not in model_cfg:
        raise ValueError("В конфиге модели нет hidden_neurons (число нейронов)")

    return MLP(
        hidden_neurons=int(model_cfg["hidden_neurons"]),
        activation=model_cfg.get("activation", "relu"),
        dropout=float(model_cfg.get("dropout", 0.2)),
        batchnorm=bool(model_cfg.get("batchnorm", True)),
    )
