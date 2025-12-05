# доступные модели, форматы, параметры
from typing import Any, Mapping, List
import importlib

from app.model_specs.catboost_classifier import get_catboost_classifier_info
from app.model_specs.logistic_regression import get_logistic_regression_info
from app.schemas import ModelClassInfo


def import_string(path: str) -> Any:
    """импорт объекта из указанного пути

    Args:
        path (str): путь к модели для импорта

    Raises:
        ImportError: если импорт не получился
        ImportError: если строка пути некорректна

    Returns:
        Any: _description_
    """
    try:
        module_path, attr_name = path.rsplit(".", maxsplit=1)
    except ValueError as exc:
        raise ImportError(f"Некорректный путь к классу: {path!r}") from exc

    try:
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            f"Не удалось импортировать {attr_name!r} из модуля {module_path!r}. "
            f"Проверь, что нужная библиотека установлена."
        ) from exc


AVAILABLE_MODEL_CLASSES = {
    info.key: info
    for info in [
        get_catboost_classifier_info(),
        get_logistic_regression_info(),
    ]
}


def get_model_class_info(key: str) -> ModelClassInfo:
    """вывод инфо по внутреннему ключу модели

    Args:
        key (str): внутренний ключ модели

    Raises:
        ValueError: если указано неправильное имя ключа, то выводим доступные

    Returns:
        ModelClassInfo: инфо о модели
    """
    try:
        return AVAILABLE_MODEL_CLASSES[key]
    except KeyError as exc:
        raise ValueError(
            f"Неизвестный класс модели: {key!r}. "
            f"Доступные ключи: {list(AVAILABLE_MODEL_CLASSES.keys())}"
        ) from exc


def resolve_model_class(key: str) -> Any:
    """получаем класс модели по ключу
    Пример:
        cls = resolve_model_class("logistic_regression")
        model = cls(**params)

    Args:
        key (str): внутренний  ключ модели
    """
    info = get_model_class_info(key)
    return import_string(info.class_path)


def list_available_model_classes() -> List[Mapping[str, Any]]:
    """получаем список словарей с инфо о доступных классах моделей"""
    result: List[Mapping[str, Any]] = []
    for info in AVAILABLE_MODEL_CLASSES.values():
        # Конвертируем param_schema, заменяя type: int на type: "int"
        serializable_schema = {}
        for param_name, param_info in info.param_schema.items():
            param_copy = dict(param_info)
            # Преобразуем класс типа в строку
            if "type" in param_copy:
                param_copy["type"] = param_copy["type"].__name__
            serializable_schema[param_name] = param_copy

        result.append(
            {
                "key": info.key,
                "display_name": info.display_name,
                "hyperparams": serializable_schema,  # Используем преобразованную схему
            }
        )
    return result
