# MNIST MLP на CUDA

Приложение для распознавания рукописных цифр многослойным перцептроном (MLP)
с обучением на видеокарте. `run_training.py` обучает модель, оценивает её на
train/val/test и собирает готовые данные для отчёта; `run_min_neurons.py`
подбирает минимальную ширину скрытого слоя.

## Что реализовано

- MNIST с фиксированным стратифицированным разбиением **54 000 / 6 000 / 10 000**
  (train / val / test); индексы кэшируются в `data/splits/`, проверяются на
  пересечения и дубликаты.
- MLP `784 -> hidden_neurons -> 10`: ровно один скрытый слой, ширина задаётся
  числом `model.hidden_neurons`; BatchNorm, ReLU (или tanh), Dropout.
- Обучение на CUDA: bf16 autocast, AdamW, `ReduceLROnPlateau`, early stopping
  по val loss, gradient clipping, воспроизводимый seed.
- Автоматическая оценка лучшего чекпоинта на train, val и test.
- Материалы для отчёта: `report.md`, `report_data.json`, `metrics.csv`,
  `curves.png`, `summary.json`, classification report, confusion matrix,
  per-class метрики и список уверенных ошибок.

Поиск минимальной ширины скрытого слоя при одном скрытом слое —
`run_min_neurons.py`, см. ниже. Эксперименты этапа 2 (20k + tanh) — в `PLAN.md`.

## Результаты базовой версии

Окружение: RTX 3070 (sm_86, 7.7 ГБ), torch 2.11.0+cu128, CUDA 12.8.
Базовая конфигурация — один скрытый слой на 512 нейронов.

| Метрика | Значение |
|---|---|
| Accuracy на test | **98.17 %** |
| Macro F1 на test | 98.16 % |
| Accuracy на val (лучшая эпоха 12) | 98.23 % |
| Accuracy на train | 99.99 % |
| Разрыв train–test | 1.82 п.п. |
| Время обучения | 11.8 с (≈0.69 с/эпоха), 17 эпох из 30 |
| Параметров | 408 074 |
| Память GPU | ≈26 МБ |

Артефакты: `runs/base_20261117-071847/`.

## Окружение

Нужен Python 3.10–3.13 (проверено на 3.12) и CUDA-сборка PyTorch.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

В `requirements.txt` версии PyTorch зафиксированы: `torch==2.11.0+cu128` и
`torchvision==0.26.0+cu128`. Обновлять их по отдельности нельзя — несовместимая
пара ломает импорт torchvision (`operator torchvision::nms does not exist`).

Пакет намеренно не устанавливается через `pip install -e .`: скрипт запускается
из корня проекта и импортирует `mnist_mlp` напрямую, поэтому никакие
`egg-info` и прочие артефакты сборки не создаются.

Проверка GPU:

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Запуск

Всё обучение и сбор отчёта — одной командой:

```bash
.venv/bin/python run_training.py
```

Скрипт читает `configs/base.yaml`, обучает модель, оценивает её на train/val/test
и создаёт новый каталог `runs/base_<timestamp>/` с полным набором артефактов.
По завершении в консоли печатаются accuracy и путь к `report.md`.

Smoke-проверка пайплайна на маленькой выборке выполняется правкой конфига:

```yaml
data:
  limit_train: 100
  limit_val: 200
training:
  epochs: 30
```

## Поиск минимального числа нейронов

```bash
.venv/bin/python run_min_neurons.py
```

Модель всегда имеет **один скрытый слой**; меняется только его ширина
(`model.hidden_neurons`). Скрипт читает `configs/min_neurons.yaml`:

1. стартует с 32 нейронов и увеличивает ширину на 1;
2. на каждой ширине обучает модель на 5 seed'ах (42–46);
3. останавливается на первой ширине, где **каждый** seed даёт
   test accuracy ≥ 97 % (`criterion: all`);
4. для найденной ширины собирает полный отчёт.

Результаты пишутся в `runs/min_neurons_<timestamp>/`: `search.md`,
`search.csv`, `accuracy_vs_neurons.png`, `search_data.json` и подкаталоги
прогонов.

Итог запуска `min_neurons_20261117-072816`:

| Нейронов | Параметров | Val acc | Test acc (mean ± std) | Test min |
|---|---|---|---|---|
| 32 | 25 514 | 96.03 % | 96.47 ± 0.11 % | 96.36 % |
| 42 | 33 484 | 96.74 % | 97.03 ± 0.18 % | 96.74 % |
| 44 | 35 078 | 96.80 % | 97.05 ± 0.14 % | 96.94 % |
| 45 | 35 875 | 96.80 % | 97.14 ± 0.15 % | 96.99 % |
| **46** | **36 672** | **96.87 %** | **97.13 ± 0.11 %** | **97.02 %** |

Минимум — **46 нейронов**: только здесь все 5 seed'ов проходят порог 97 %.
При 45 нейронах средняя уже 97.14 %, но один seed даёт 96.99 %, поэтому
критерий `all` не выполняется. Разница между соседними ширинами (±0.1–0.2 п.п.)
сопоставима с разбросом по seed'ам, поэтому 46 — минимум для данного критерия
и фиксированных seed'ов, а не гарантированная теоретическая граница.

## Конфиг

`configs/base.yaml`:

- данные: `root`, `val_ratio`, `split_seed`, `batch_size`, `num_workers`,
  `limit_train`, `limit_val`;
- модель: `hidden_neurons` (число нейронов единственного скрытого слоя),
  `activation` (`relu`/`tanh`/`gelu`), `dropout`, `batchnorm`;
- обучение: `epochs`, `lr`, `weight_decay`, `patience`, `min_delta`, `amp`,
  `amp_dtype`, `grad_clip`, планировщик;
- каталог результатов: `output.dir`.

## Артефакты запуска

```
runs/base_<timestamp>/
├── best.pt, last.pt        чекпоинты (best — по val loss)
├── metrics.csv             метрики по эпохам: loss, accuracy, lr, время, GPU
├── curves.png              кривые обучения train/val
├── summary.json            сводка обучения
├── report.md               готовый Markdown-отчёт
├── report_data.json        все числа одним файлом
├── data_split.json         размеры и fingerprint разбиения
└── eval_train|val|test/
    ├── metrics.json        accuracy, F1, confusion matrix, per-class
    ├── report.txt          classification report
    ├── per_class.csv       precision/recall/F1 по цифрам
    ├── confusion_matrix.png
    └── top_errors.csv      самые уверенные ошибки
```

## Структура

```
run_training.py          обучение одной конфигурации + report.md
run_min_neurons.py       поиск минимальной ширины скрытого слоя, 5 seed'ов
configs/base.yaml        гиперпараметры
configs/min_neurons.yaml параметры поиска минимальной ширины
mnist_mlp/data.py        MNIST, стратифицированное разбиение, DataLoader
mnist_mlp/model.py       MLP, инициализация, сборка по конфигу
mnist_mlp/train.py       цикл обучения, AMP, early stopping, чекпоинты
mnist_mlp/analysis.py    метрики, confusion matrix, ошибки
mnist_mlp/report.py      сборка report.md и report_data.json
tests/                   быстрые тесты
runs/<run>/              результаты запуска
```

## Тесты

```bash
.venv/bin/python -m pytest -q
```
