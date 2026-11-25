#!/usr/bin/env python3
"""Лабораторная работа: классификация рукописных цифр MNIST.

Основная логика здесь намеренно короткая:
загрузка данных -> создание MLP -> обучение -> сравнение результатов.
Настройка CUDA находится отдельно в cuda_setup.py.

Запуск:
    python mnist_lab.py

Для быстрой проверки:
    python mnist_lab.py --quick
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
import matplotlib
import numpy as np
import tensorflow as tf
from keras import layers

from cuda_setup import device_info, use_cuda

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")
SEED = 42


def load_mnist() -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Загружает MNIST и делит исходные 60 000 изображений на 54k/6k."""
    (x, y), (x_test, y_test) = keras.datasets.mnist.load_data()

    x = x.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Один и тот же seed даёт одинаковое разбиение при каждом запуске.
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(x))
    train_indices = indices[:54_000]
    val_indices = indices[54_000:]

    return (
        (x[train_indices], y[train_indices]),
        (x[val_indices], y[val_indices]),
        (x_test, y_test),
    )


def create_model(hidden_neurons: int, activation: str) -> keras.Model:
    """Создаёт MLP: 784 -> hidden_neurons -> 10."""
    model = keras.Sequential(
        [
            layers.Input(shape=(28, 28)),
            layers.Flatten(),
            layers.Dense(hidden_neurons, activation=activation),
            layers.Dropout(0.2),
            layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(
    name: str,
    hidden_neurons: int,
    activation: str,
    train_data: tuple[np.ndarray, np.ndarray],
    val_data: tuple[np.ndarray, np.ndarray],
    test_data: tuple[np.ndarray, np.ndarray],
    epochs: int,
    seed: int = SEED,
) -> dict[str, float | int | str]:
    """Обучает одну модель и возвращает основные показатели."""
    # Seed задаёт инициализацию сети и порядок перемешивания.
    keras.utils.set_random_seed(seed)
    model = create_model(hidden_neurons, activation)
    output_dir = RESULTS_DIR / name
    output_dir.mkdir(parents=True, exist_ok=True)

    history = model.fit(
        train_data[0],
        train_data[1],
        validation_data=val_data,
        epochs=epochs,
        batch_size=256,
        shuffle=True,
        callbacks=[
            # Контроль переобучения по validation loss.
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=4, restore_best_weights=True
            ),
        ],
        verbose=2,
    )

    train_loss, train_accuracy = model.evaluate(*train_data, verbose=0)
    val_loss, val_accuracy = model.evaluate(*val_data, verbose=0)
    test_loss, test_accuracy = model.evaluate(*test_data, verbose=0)
    model.save(output_dir / "model.keras")
    save_history(history, output_dir / "learning_curves.png", name)

    result: dict[str, float | int | str] = {
        "name": name,
        "activation": activation,
        "hidden_neurons": hidden_neurons,
        "train_size": len(train_data[0]),
        "epochs": len(history.history["loss"]),
        "train_accuracy": float(train_accuracy),
        "val_accuracy": float(val_accuracy),
        "test_accuracy": float(test_accuracy),
        "overfitting_gap": float(train_accuracy - val_accuracy),
        "parameters": model.count_params(),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def find_minimum_width(
    data: tuple[tuple[np.ndarray, np.ndarray], ...], epochs: int
) -> tuple[int, list[dict]]:
    """Ищет минимум бинарным поиском на отрезке 40..50.

    Ширина считается подходящей, если все пять seed дают validation accuracy
    не ниже 97 %. Test используется только в итоговом эксперименте.
    """
    train_data, val_data, test_data = data
    seeds = [42, 43, 44, 45, 46]
    results: list[dict] = []
    left, right = 40, 50
    best = right

    while left <= right:
        neurons = (left + right) // 2
        run_results = []
        for seed in seeds:
            result = train_model(
                f"search_{neurons}_seed_{seed}", neurons, "relu",
                train_data, val_data, test_data, epochs, seed
            )
            result["search_seed"] = seed
            run_results.append(result)
            results.append(result)

        values = [float(item["val_accuracy"]) for item in run_results]
        minimum = min(values)
        average = sum(values) / len(values)
        passed = minimum >= 0.97
        print(
            f"{neurons} нейронов, seeds {seeds}: "
            f"val mean={average:.4f}, val min={minimum:.4f}"
        )

        if passed:
            best = neurons
            right = neurons - 1
        else:
            left = neurons + 1

    return best, results


def save_history(history: keras.callbacks.History, path: Path, title: str) -> None:
    """Сохраняет простой график loss и accuracy."""
    epochs = range(1, len(history.history["loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, history.history["loss"], label="train")
    axes[0].plot(epochs, history.history["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history.history["accuracy"], label="train")
    axes[1].plot(epochs, history.history["val_accuracy"], label="validation")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def save_results(results: list[dict], selected_neurons: int) -> None:
    """Сохраняет таблицу результатов для отчёта."""
    with (RESULTS_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    summary = {
        "gpu": device_info(),
        "selected_neurons": selected_neurons,
        "results": results,
    }
    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="2 эпохи для проверки кода")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    use_cuda()
    data = load_mnist()
    train_data, val_data, test_data = data
    epochs = 2 if args.quick else 30

    print(f"train: {len(train_data[0])}, validation: {len(val_data[0])}, test: {len(test_data[0])}")

    selected_neurons, search_results = find_minimum_width(data, epochs)
    print(f"Выбрано нейронов: {selected_neurons}")

    # Эксперимент 1: полный обучающий набор и ReLU.
    baseline = train_model(
        "full_relu", selected_neurons, "relu",
        train_data, val_data, test_data, epochs, SEED
    )

    # Эксперимент 2: 20 000 случайных объектов и tanh.
    rng = np.random.default_rng(SEED)
    subset_indices = rng.choice(len(train_data[0]), 20_000, replace=False)
    subset = (train_data[0][subset_indices], train_data[1][subset_indices])
    subset_tanh = train_model(
        "subset_20k_tanh", selected_neurons, "tanh",
        subset, val_data, test_data, epochs, SEED
    )

    save_results(search_results + [baseline, subset_tanh], selected_neurons)
    print("Результаты сохранены в каталоге results/")


if __name__ == "__main__":
    main()
