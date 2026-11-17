# MNIST MLP CUDA

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
`run_min_neurons.py`, этап 2 (20 000 объектов) — `run_stage2.py`, см. ниже.

## Результаты моделей (46 нейронов)

Окружение: RTX 3070 (sm_86, 7.7 ГБ), torch 2.11.0+cu128, CUDA 12.8.
Ширина скрытого слоя зафиксирована по результатам поиска: **46 нейронов**.
ReLU и tanh вынесены в отдельные конфиги и обучены независимо.

| Метрика | ReLU (`configs/relu.yaml`) | tanh (`configs/tanh.yaml`) |
|---|---|---|
| Accuracy на test | **97.08 %** | 96.21 % |
| Macro F1 на test | 97.05 % | 96.17 % |
| Accuracy на val (лучшая эпоха) | 96.92 % (25) | 95.70 % (26) |
| Accuracy на train | 98.93 % | 98.56 % |
| Разрыв train–test | 1.85 п.п. | 2.35 п.п. |
| Время обучения | 23.3 с, 30 эпох | 20.6 с, 30 эпох |
| Параметров | 36 672 | 36 672 |
| Память GPU | ≈18 МБ | ≈18 МБ |

ReLU-модель проходит порог 97 %, tanh при той же ширине — нет (96.21 %).
На последней эпохе доля насыщенных tanh-нейронов (|tanh| > 0.99) составила
около 37 %, что указывает на ограниченную ёмкость tanh при 46 нейронах.
Это результат обучения при фиксированной ширине; отдельный подбор ширины
для tanh не выполнялся.

Артефакты: `runs/relu_20261117-182122/`, `runs/tanh_20261117-182152/`.

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

Модель ReLU:

```bash
.venv/bin/python run_training.py
```

Модель tanh:

```bash
.venv/bin/python run_training_tanh.py
```

Оба скрипта читают свой конфиг (`configs/relu.yaml` и `configs/tanh.yaml`),
обучают модель, оценивают её на train/val/test и создают новый каталог
`runs/relu_<timestamp>/` или `runs/tanh_<timestamp>/` с полным набором
артефактов. По завершении в консоли печатаются accuracy и путь к `report.md`.

Конфиги отличаются только активацией; ширина скрытого слоя в обоих — 46.

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

## Этап 2: обучение на 20 000 объектов

```bash
.venv/bin/python run_stage2.py
```

Скрипт читает `configs/stage2_tanh.yaml` (E2) и `configs/stage2_relu.yaml`
(E3). Оба конфига используют `data.train_subset: 20000`: из train-пула 54 000
случайно без замены отбираются 20 000 объектов. Подмножество зависит от seed
прогона, поэтому 3 seed'а (42–44) дают 3 разных набора. Val 6 000 и test
10 000 — те же, что на этапе 1. Ширина скрытого слоя не подбирается заново:
во всех прогонах этапа 2 используется зафиксированная **46** — та же, что
в E1/E4.

| Эксперимент | Активация | Train | Test acc (mean ± std) | Test min | Val acc |
|---|---|---|---|---|---|
| E2 | tanh | 20 000 | 94.24 ± 0.22 % | 93.98 % | 93.73 % |
| E3 | relu | 20 000 | 95.58 ± 0.16 % | 95.40 % | 95.24 % |

Сравнение с этапом 1 при той же ширине 46:

| Условие | Train | Активация | Test acc |
|---|---|---|---|
| E1 | 54 000 | relu | 97.08 % |
| E4 | 54 000 | tanh | 96.21 % |
| E3 | 20 000 | relu | 95.58 ± 0.16 % |
| E2 | 20 000 | tanh | 94.24 ± 0.22 % |

При 46 нейронах сокращение выборки до 20k стоит ≈1.5 п.п. для ReLU и ≈2.0 п.п.
для tanh; tanh проигрывает ReLU 1.34 п.п. Обе модели на 20k ниже порога 97 % —
это ожидаемо: ширина 46 подобрана как минимум для 54k, а по заданию для этапа 2
она не пересматривается.

Артефакты: `runs/stage2_20261117-182941/`.

## Сводка E1–E5

```bash
.venv/bin/python run_summary.py
```

Скрипт собирает уже выполненные прогоны и создаёт `runs/summary_<timestamp>/`:

- `summary.md` — сводная таблица E1–E5, сходимость и разбивка по прогонам;
- `summary.csv` — агрегаты по экспериментам;
- `runs.csv` — все прогоны;
- `convergence.csv` — эпохи достижения val 95 % и 98 %;
- `learning_curves.png` — loss и accuracy по эпохам для E1–E4 (seed 42);
- `summary_accuracy.png` — test accuracy с разбросом и val accuracy;
- `summary_data.json` — все числа и пути.

| Эксперимент | Train | Активация | Test acc | Val acc | Train acc | Зазор |
|---|---|---|---|---|---|---|
| E1 | 54 000 | relu | 97.08 % | 96.92 % | 98.94 % | 1.86 п.п. |
| E2 | 20 000 | tanh | 94.24 ± 0.22 % | 93.73 ± 0.10 % | 98.80 % | 4.57 п.п. |
| E3 | 20 000 | relu | 95.58 ± 0.16 % | 95.24 ± 0.23 % | 99.09 % | 3.51 п.п. |
| E4 | 54 000 | tanh | 96.21 % | 95.70 % | 98.56 % | 2.35 п.п. |
| E5 | 54 000 | relu | 97.13 ± 0.11 % | 96.87 ± 0.12 % | 99.08 % | 1.95 п.п. |

Сходимость: на 54k ReLU доходит до val 95 % за 4 эпохи, tanh — за 13;
на 20k ReLU — за 19 эпох, tanh не достигает 95 % ни на одном seed. До val
98 % не доходит ни один эксперимент.

## Конфиг

`configs/relu.yaml` и `configs/tanh.yaml` (отличаются только `model.activation`):

- данные: `root`, `val_ratio`, `split_seed`, `batch_size`, `num_workers`,
  `limit_train`, `limit_val`, `train_subset` (null или 20 000);
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
run_training.py          обучение ReLU-модели + report.md
run_training_tanh.py     обучение tanh-модели + report.md
run_stage2.py            этап 2: 20k, tanh/ReLU, 3 seed'а
run_summary.py           сводка E1–E5: таблицы, сходимость, графики
run_min_neurons.py       поиск минимальной ширины скрытого слоя, 5 seed'ов
configs/relu.yaml        гиперпараметры ReLU-модели
configs/tanh.yaml        гиперпараметры tanh-модели
configs/stage2_tanh.yaml этап 2: 20k + tanh
configs/stage2_relu.yaml этап 2: 20k + ReLU
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
