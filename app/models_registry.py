# доступные модели, форматы, параметры

from dataclasses import dataclass
from typing import Any, Dict, Mapping, List
import importlib


@dataclass
class ModelClassInfo:
    """описание доступного класса модели для обучения
    key: ключ, по которому обращаемся к модели
    display_name: отображаемое в ui имя для читаемости
    class_path: путь до класса
    default_params: дефольные параметры создания модели, если не указаны пользователем
    param_schema: описание настраиваемых гиперпараметров
    """

    key: str
    display_name: str
    class_path: str
    default_params: Dict[str, Any]
    param_schema: Dict[str, Dict[str, Any]]


"""
    Импортирует объект по строке вида 'module.submodule.ClassName'.

    Пример:
        cls = import_string("sklearn.linear_model.LogisticRegression")

    :raises ImportError: если импорт не удался или строка пути некорректна.
    """


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
    # модель 1 - градиентный бустинг CatBoostClassifier
    "catboost_classifier": ModelClassInfo(
        key="catboost_classifier",
        display_name="CatBoostClassifier",
        class_path="catboost.CatBoostClassifier",
        default_params={
            "iterations": 200,
            "learning_rate": 0.1,
            "depth": 6,
            "verbose": False,
        },
        param_schema={
            "iterations": {
                "type": int,
                "min": 1,
                "max": 5000,
                "description": "Количество итераций бустинга (деревьев).",
            },
            "learning_rate": {
                "type": float,
                "min": 1e-4,
                "max": 1.0,
                "description": "Шаг обучения (learning rate).",
            },
            "depth": {
                "type": int,
                "min": 1,
                "max": 16,
                "description": "Глубина деревьев.",
            },
        },
    ),
    # модель 2 - логистическая регрессия LogisticRegression
    "logistic_regression": ModelClassInfo(
        key="logistic_regression",
        display_name="LogisticRegression",
        class_path="sklearn.linear_model.LogisticRegression",
        default_params={
            "C": 1.0,
            "max_iter": 100,
            "penalty": "l2",
            "solver": "lbfgs",
        },
        param_schema={
            "C": {
                "type": float,
                "min": 1e-6,
                "max": 1e6,
                "description": "Обратный коэффициент регуляризации (чем больше, тем слабее регуляризация).",
            },
            "max_iter": {
                "type": int,
                "min": 10,
                "max": 10000,
                "description": "Максимальное число итераций оптимизатора.",
            },
            "penalty": {
                "type": str,
                "choices": ["l2", "none"],
                "description": "Тип регуляризации (ограничиваемся 'l2' и 'none' для простоты).",
            },
            "solver": {
                "type": str,
                "choices": ["lbfgs", "liblinear", "saga"],
                "description": "Алгоритм оптимизации.",
            },
        },
    ),
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
