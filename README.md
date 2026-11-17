# MNIST MLP на CUDA

Приложение для распознавания рукописных цифр многослойным перцептроном (MLP)
с обучением на видеокарте. Одна точка входа — `run_training.py`: она обучает
модель, оценивает её на train/val/test и собирает готовые данные для отчёта.

## Что реализовано

- MNIST с фиксированным стратифицированным разбиением **54 000 / 6 000 / 10 000**
  (train / val / test); индексы кэшируются в `data/splits/`, проверяются на
  пересечения и дубликаты.
- MLP `784 -> 512 -> 256 -> 10`: BatchNorm, ReLU (или tanh), Dropout.
- Обучение на CUDA: bf16 autocast, AdamW, `ReduceLROnPlateau`, early stopping
  по val loss, gradient clipping, воспроизводимый seed.
- Автоматическая оценка лучшего чекпоинта на train, val и test.
- Материалы для отчёта: `report.md`, `report_data.json`, `metrics.csv`,
  `curves.png`, `summary.json`, classification report, confusion matrix,
  per-class метрики и список уверенных ошибок.

Эксперименты E2–E5 (20k + tanh, контрольные прогоны, минимальная ширина) —
следующий этап, см. `PLAN.md`.

## Результаты стартовой версии

Окружение: RTX 3070 (sm_86, 7.7 ГБ), torch 2.11.0+cu128, CUDA 12.8.

| Метрика | Значение |
|---|---|
| Accuracy на test | **98.59 %** |
| Macro F1 на test | 98.58 % |
| Accuracy на val (лучшая эпоха 15) | 98.62 % |
| Accuracy на train | 99.98 % |
| Разрыв train–test | 1.39 п.п. |
| Время обучения | 16.2 с (≈0.81 с/эпоха), 20 эпох из 30 |
| Параметров | 537 354 |
| Память GPU | ≈29 МБ |

Артефакты: `runs/base_20261117-062846/`.

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

## Подбор конфигурации E1

```bash
.venv/bin/python run_e1_selection.py
```

Скрипт читает `configs/e1_search.yaml` и работает в три шага:

1. **Скрининг** 15 конфигураций на одном seed: ширины скрытых слоёв,
   dropout, learning rate, weight decay.
2. **Подтверждение** top-3 по validation на seed 42/43/44.
3. **Выбор** по средней val accuracy (при равенстве — по val loss);
   test считается только для победителя, один раз.

Результаты пишутся в `runs/e1_search_<timestamp>/`: `comparison.md`,
`comparison.csv`, `val_accuracy.png`, `search_data.json` и подкаталоги прогонов.

Итог запуска `e1_search_20261117-064351`:

| Конфигурация | Параметров | Val accuracy | Test accuracy |
|---|---|---|---|
| w1024-512 (победитель) | 1 336 842 | 98.59 ± 0.09 % | 98.30 % |
| w768-384 | 904 330 | 98.56 ± 0.10 % | — |
| d0.3 | 537 354 | 98.53 ± 0.25 % | — |
| w512-256 (базовая) | 537 354 | 98.62 % (1 seed) | 98.59 % |

Разница между финалистами по validation — 0.03–0.06 п.п., что меньше
стандартной ошибки оценки (≈0.17 п.п. на test). Поэтому выбор по validation
может не совпасть с test: в этом запуске победитель по val уступил базовой
модели на test. Это ожидаемый winner's curse при отборе из многих конфигураций;
для отчёта важно приводить разброс и не трактовать десятые доли процента как
значимое превосходство.

## Конфиг

`configs/base.yaml`:

- данные: `root`, `val_ratio`, `split_seed`, `batch_size`, `num_workers`,
  `limit_train`, `limit_val`;
- модель: `hidden`, `activation` (`relu`/`tanh`/`gelu`), `dropout`, `batchnorm`;
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
run_e1_selection.py      подбор E1: скрининг, подтверждение, сравнение
configs/base.yaml        гиперпараметры
configs/e1_search.yaml   кандидаты для подбора E1
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
