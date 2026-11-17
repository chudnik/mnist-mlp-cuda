"""Сборка Markdown-отчёта и сводного JSON из результатов обучения.

``build_report`` вызывается из run_training.py после оценки модели на
train/val/test. На вход он получает каталог запуска, словарь результатов
(по выборкам) и конфиг, а на выходе создаёт:

  * ``report.md`` — человекочитаемый отчёт для преподавателя;
  * ``report_data.json`` — те же данные машинночитаемо (используется
    сводными скриптами run_summary.py и run_stage2.py).

Модуль ничего не считает сам: все метрики уже посчитаны в analysis.py,
здесь они только форматируются в таблицы.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import load_json, save_json


def _pct(value: float) -> str:
    """Форматирует долю (0..1) как процент с двумя знаками: 0.9708 -> '97.08 %'."""
    return f"{value * 100:.2f} %"


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Собирает Markdown-таблицу и возвращает её строками.

    ``headers`` — названия колонок, ``rows`` — уже отформатированные
    значения. Разделитель ``---`` обязателен для валидной Markdown-таблицы.
    Возвращаются строки (а не одна строка), чтобы вызывающий код мог
    просто добавить их к общему списку ``lines``.
    """
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def build_report(
        run_dir: str | Path,
        results: dict[str, dict[str, Any]],
        config: dict[str, Any],
) -> Path:
    """Создаёт report.md и report_data.json для одного запуска.

    Аргументы:
        run_dir: каталог запуска, где уже лежат summary.json и артефакты
            оценки (eval_train/, eval_val/, eval_test/).
        results: словарь {"train"/"val"/"test": метрики}; каждая метрика —
            то, что вернул ``evaluate_split`` (accuracy, macro_f1,
            weighted_f1, per_class, confusion_matrix, num_samples).
        config: конфиг запуска (нужен для секций model и training).

    Из отчёта можно узнать:
        * размеры выборок и fingerprint разбиения — доказательство, что
          test не участвовал в обучении;
        * архитектуру и число параметров;
        * сколько эпох прошло, была ли ранняя остановка;
        * accuracy/F1 на train, val и test и разрыв train–test
          (индикатор переобучения);
        * precision/recall/F1 по каждой цифре на test.

    Возвращает путь к созданному report.md.
    """
    run_dir = Path(run_dir)
    summary = load_json(run_dir / "summary.json")
    meta = summary["data"]
    model_cfg = config["model"]
    training_cfg = config["training"]
    test = results["test"]

    # Разрыв train–test в процентных пунктах: чем больше, тем сильнее
    # переобучение. Считается по accuracy, а не по loss, потому что она
    # интерпретируется проще.
    gap = results["train"]["accuracy"] - test["accuracy"]
    hidden = str(model_cfg["hidden_neurons"])

    lines: list[str] = [
        "# Отчёт об обучении MLP на MNIST",
        "",
        f"Запуск: `{run_dir.name}`  ",
        f"Устройство: {summary['device']}",
        "",
        "## Данные",
        "",
        f"- train: {meta['train_size']}",
        f"- validation: {meta['val_size']}",
        f"- test: {meta['test_size']}",
        f"- val_ratio: {meta['val_ratio']}, split seed: {meta['split_seed']}",
        f"- fingerprint разбиения: `{meta['fingerprint']}`",
        "",
        "## Модель",
        "",
        f"- архитектура: 784-{hidden}-10",
        f"- активация: {model_cfg['activation']}",
        f"- dropout: {model_cfg['dropout']}, batchnorm: {model_cfg['batchnorm']}",
        f"- параметров: {summary['params']:,}",
        "",
        "## Обучение",
        "",
        f"- эпох выполнено: {summary['epochs_ran']} (лимит {training_cfg['epochs']})",
        f"- лучшая эпоха: {summary['best_epoch']}",
        f"- early stopping: {'да' if summary['early_stopped'] else 'нет'}",
        f"- время: {summary['seconds']:.1f} с "
        f"(≈{summary['seconds_per_epoch']:.2f} с/эпоха)",
        f"- оптимизатор: AdamW, lr={training_cfg['lr']}, "
        f"weight_decay={training_cfg['weight_decay']}",
        f"- AMP: {training_cfg['amp']} ({training_cfg['amp_dtype']}), "
        f"batch={meta.get('batch_size', config['data']['batch_size'])}",
        "",
        "## Результаты",
        "",
    ]

    # Основная таблица: три выборки и четыре метрики.
    lines += _table(
        ["Выборка", "Accuracy", "Macro F1", "Weighted F1", "Примеров"],
        [
            [
                split,
                _pct(results[split]["accuracy"]),
                _pct(results[split]["macro_f1"]),
                _pct(results[split]["weighted_f1"]),
                str(results[split]["num_samples"]),
            ]
            for split in ("train", "val", "test")
        ],
    )
    lines += [
        "",
        f"Разрыв train–test: **{gap * 100:.2f} п.п.**  ",
        f"Лучшая val accuracy: **{_pct(summary['best_val_acc'])}** "
        f"(val loss {summary['best_val_loss']:.4f})",
        "",
        "## Качество по классам (test)",
        "",
    ]
    # Детализация по цифрам: видно, какие классы путаются чаще (обычно 4/9, 3/5, 7/9).
    lines += _table(
        ["Цифра", "Precision", "Recall", "F1", "Support"],
        [
            [
                digit,
                _pct(values["precision"]),
                _pct(values["recall"]),
                _pct(values["f1"]),
                str(values["support"]),
            ]
            for digit, values in test["per_class"].items()
        ],
    )
    lines += [
        "",
        "## Артефакты",
        "",
        "- `best.pt`, `last.pt` — чекпоинты",
        "- `metrics.csv` — метрики по эпохам (loss, accuracy, lr, время, GPU)",
        "- `curves.png` — кривые обучения",
        "- `summary.json` — сводка обучения",
        "- `eval_train/`, `eval_val/`, `eval_test/` — метрики, "
        "classification report, confusion matrix, ошибки",
        "- `report_data.json` — все числа одним файлом",
        "",
        "## Примечание",
        "",
        "Test использовался только для финальной оценки выбранного по validation "
        "чекпоинта; гиперпараметры по test не подбирались.",
        "",
    ]

    report_path = run_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    # report_data.json дублирует всё в машинночитаемом виде: из него потом
    # собираются сводные таблицы E1–E5.
    save_json(
        {"config": config, "summary": summary, "results": results},
        run_dir / "report_data.json",
    )
    return report_path
