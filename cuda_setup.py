"""Техническая настройка TensorFlow для работы на CUDA.

Эта часть не относится к алгоритму лабораторной работы. Её можно не разбирать
при защите: она только проверяет наличие видеокарты и ограничивает TensorFlow
по объёму используемой памяти.
"""

import tensorflow as tf


def use_cuda() -> None:
    """Включает GPU и сообщает, какое устройство будет использоваться."""
    gpus = tf.config.list_physical_devices("GPU")

    if not gpus:
        raise RuntimeError(
            "CUDA GPU не найдена. По условию лабораторной обучение должно "
            "выполняться на видеокарте."
        )

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    print(f"Используется CUDA GPU: {gpus[0].name}")


def device_info() -> list[str]:
    """Возвращает список доступных GPU для сохранения в отчёте."""
    return [gpu.name for gpu in tf.config.list_physical_devices("GPU")]
