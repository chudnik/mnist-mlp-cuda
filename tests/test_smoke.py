"""Быстрые тесты без загрузки данных и без GPU."""

from __future__ import annotations

import numpy as np
import torch

from mnist_mlp.analysis import compute_metrics
from mnist_mlp.data import Split, stratified_split
from mnist_mlp.model import MLP, build_model


def test_stratified_split_is_disjoint_and_balanced() -> None:
    targets = np.repeat(np.arange(10), 100)
    split = stratified_split(targets, val_ratio=0.1, seed=42)

    assert isinstance(split, Split)
    assert split.train.size == 900
    assert split.val.size == 100
    assert np.intersect1d(split.train, split.val).size == 0

    val_labels = targets[split.val]
    counts = np.bincount(val_labels, minlength=10)
    assert counts.tolist() == [10] * 10


def test_stratified_split_is_reproducible() -> None:
    targets = np.repeat(np.arange(10), 50)
    first = stratified_split(targets, 0.2, seed=7)
    second = stratified_split(targets, 0.2, seed=7)
    assert np.array_equal(first.train, second.train)
    assert np.array_equal(first.val, second.val)
    assert first.fingerprint == second.fingerprint


def test_mlp_output_shape() -> None:
    model = MLP(hidden=(16, 8), dropout=0.0)
    model.eval()
    output = model(torch.randn(4, 1, 28, 28))
    assert output.shape == (4, 10)


def test_tanh_model_builds_and_runs() -> None:
    model = build_model(
        {
            "hidden": [32, 16],
            "activation": "tanh",
            "dropout": 0.1,
            "batchnorm": True,
        }
    )
    model.eval()
    output = model(torch.randn(3, 28, 28))
    assert output.shape == (3, 10)
    assert torch.isfinite(output).all()


def test_mlp_flattens_input() -> None:
    model = MLP(hidden=(8,), dropout=0.0)
    model.eval()
    flat = torch.flatten(torch.ones(2, 1, 28, 28), start_dim=1)
    assert torch.allclose(model(flat), model(torch.ones(2, 1, 28, 28)))


def test_compute_metrics_perfect_predictions() -> None:
    labels = np.arange(10)
    probs = np.eye(10, dtype=np.float32)
    metrics = compute_metrics(labels, probs)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["num_samples"] == 10
    assert np.asarray(metrics["confusion_matrix"]).trace() == 10
