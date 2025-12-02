# выбор класса, валидация гиперпараметров, train/predict/retrain/delet

from typing import Any, Dict, Sequence, Optional, List, Tuple

import logging
import os 

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from app import models_registry
from app.storage import storage, ModelMetadata

from clearml import Task

logger = logging.getLogger("ml_service")

def _start_clearml_task(
    action: str,
    model_class_key: str,
    extra_config: Optional[Dict[str, Any]] = None,
):
    """
    Создаёт ClearML Task для обучения/переобучения, если ClearML доступен и настроен.
    Возвращает объект Task или None, если трекинг недоступен.
    """
    if Task is None:
        return None

    try:
        task = Task.init(
            project_name=os.getenv("CLEARML_PROJECT", "HW_MLOps"),
            task_name=f"{action}_{model_class_key}",
            task_type=Task.TaskTypes.training,
        )
    except Exception as exc:
        logger.warning("Не удалось инициализировать ClearML Task: %s", exc)
        return None

    config: Dict[str, Any] = {"model_class_key": model_class_key}
    if extra_config:
        config.update(extra_config)

    try:
        # сохраняем конфиг эксперимента (гиперпараметры, размеры данных и т.п.)
        task.connect(config, name="config")
    except Exception as exc:
        logger.warning("Не удалось отправить конфиг в ClearML: %s", exc)

    return task


def _to_primitive(value: Any) -> Any:
    """вспомогательная функция, приводит значение к типу, который сериализуется в json

    Args:
        value (Any): значение

    Returns:
        Any: приведенное значение
            - int, float, str, bool, None — оставляет как есть
            - всё остальное пытается привести к float, иначе к str()
    """
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    try:
        return float(value)
    except Exception:
        return str(value)


def _validate_X_and_y(
    X: Sequence[Sequence[Any]],
    y: Sequence[Any],
) -> int:
    """проверяет базовую консистентность X и y

    Args:
        X (Sequence[Sequence[Any]]): признаки
        y (Sequence[Any]): таргет

    Raises:
        ValueError: пустая обучеющая выборка
        ValueError: длины признаков и таргета не совпадают
        ValueError: разное количество признаков у объектов

    Returns:
        int: _description_
    """
    n_samples = len(X)
    if n_samples == 0:
        raise ValueError("Пустая обучающая выборка: len(X) == 0.")

    if len(y) != n_samples:
        raise ValueError(
            f"len(y)={len(y)} не совпадает с len(X)={n_samples}. "
            "Каждой строке X должна соответствовать ровно одна метка y."
        )

    n_features = len(X[0])
    for i, row in enumerate(X):
        if len(row) != n_features:
            raise ValueError(
                "Все объекты X должны иметь одинаковое число признаков. "
                f"Строка с индексом {i} имеет {len(row)} признаков, "
                f"ожидается {n_features}."
            )

    return n_features


def _validate_and_merge_hyperparams(
    model_class_key: str, user_params: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """вспомогательная функция, валидирует пользовательские гиперпараметры и возвращает итоговый словарь

    Args:
        model_class_key (str): внутренний ключ модели
        user_params (Optional[Dict[str, Any]]): параметры модели, введенные пользователем

    Raises:
        ValueError: при неподдержке гиперпараметров моделью
        ValueError: при невозможности приведения гиперпараметра к правильному типа
        ValueError: при недопустимом значении гиперпараметра (больше / меньше допустимого)
        ValueError: при попытке ввести неподдерживаемые гиперпараметры

    Returns:
        Dict[str, Any]: итоговый словарь с гиперпараметрами
    """
    info = models_registry.get_model_class_info(model_class_key)
    schema = info.param_schema
    final_params: Dict[str, Any] = dict(info.default_params)

    if user_params is None:
        return final_params

    for name, raw_value in user_params.items():
        if name not in schema:
            raise ValueError(
                f"Гиперпараметр {name!r} не поддерживается для модели {model_class_key!r}. "
                f"Разрешённые параметры: {list(schema.keys())}"
            )

        param_info = schema[name]
        expected_type = param_info.get("type", type(raw_value))

        # Приводим тип (int/float/str)
        try:
            value = expected_type(raw_value)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                f"Не удалось привести параметр {name!r} к типу {expected_type.__name__}: {raw_value!r}"
            ) from exc

        # Проверяем диапазон для чисел
        if isinstance(value, (int, float)):
            min_val = param_info.get("min")
            max_val = param_info.get("max")
            if min_val is not None and value < min_val:
                raise ValueError(
                    f"Значение {name!r}={value} меньше минимально допустимого ({min_val})."
                )
            if max_val is not None and value > max_val:
                raise ValueError(
                    f"Значение {name!r}={value} больше максимально допустимого ({max_val})."
                )

        # Проверяем список допустимых значений (для строк и не только)
        choices = param_info.get("choices")
        if choices is not None and value not in choices:
            raise ValueError(
                f"Значение {name!r}={value!r} не входит в список допустимых {choices}."
            )

        final_params[name] = value

    return final_params


def _infer_feature_indices(
    X: Sequence[Sequence[Any]],
    feature_types: Optional[Sequence[Optional[str]]],
) -> Tuple[List[int], List[int]]:
    """определяем, какие признаки количественные, а какие категориальные

    Args:
        X (Sequence[Sequence[Any]]): признаки
        feature_types (Optional[Sequence[Optional[str]]]): опциональный список типов для каждого признака
            - 'num', 'numeric', 'float', 'int' => числовой
            - 'cat', 'categorical', 'str', 'string', 'object' => категориальный
            - None => считаем числовым (по умолчанию)

    Если feature_types == None:
        - смотрим на типы значений в X (по всем строкам):
            - если все значения либо int/float/bool/None => считаем признак числовым
            - если встречается str или что-то ещё => категориальный

    Raises:
        ValueError: длина типов признаков не совпадает с количеством признаков
        ValueError: неизвестный тип признака в feature_types
        ValueError: признаки пустые

    Returns:
        Tuple[List[int], List[int]]: списки индексов признаков
    """
    if not X:
        raise ValueError("X пустой при вызове _infer_feature_indices.")

    n_features = len(X[0])

    # Явно заданные типы признаков
    if feature_types is not None:
        if len(feature_types) != n_features:
            raise ValueError(
                f"Длина feature_types ({len(feature_types)}) не совпадает с числом признаков ({n_features})."
            )

        numeric_indices: List[int] = []
        cat_indices: List[int] = []

        for idx, t in enumerate(feature_types):
            if t is None:
                numeric_indices.append(idx)
                continue

            s = str(t).lower()
            if s in ("num", "numeric", "number", "float", "int"):
                numeric_indices.append(idx)
            elif s in ("cat", "categorical", "category", "string", "str", "object"):
                cat_indices.append(idx)
            else:
                raise ValueError(
                    f"Неизвестный тип признака {t!r} для колонки {idx}. "
                    "Используйте 'num'/'numeric' или 'cat'/'categorical'."
                )

        return numeric_indices, cat_indices

    # Автоопределение по типам значений
    numeric_indices = []
    cat_indices = []

    for col_idx in range(n_features):
        is_categorical = False
        for row in X:
            val = row[col_idx]
            if val is None:
                # Пропуски не влияют на тип; смотрим дальше
                continue
            # int/float/bool → считаем числовым
            if isinstance(val, (int, float, bool)):
                continue
            # Всё остальное (str и т.п.) считаем категориальным
            is_categorical = True
            break

        if is_categorical:
            cat_indices.append(col_idx)
        else:
            numeric_indices.append(col_idx)

    return numeric_indices, cat_indices


def _build_logistic_pipeline(
    params: Dict[str, Any],
    numeric_indices: List[int],
    cat_indices: List[int],
) -> Any:
    """строит sklearn Pipeline для логистической регрессии
        [ColumnTransformer(num -> StandardScaler, cat -> OneHotEncoder)] -> LogisticRegression

    Args:
        params (Dict[str, Any]): гиперпараметры для LogisticRegression
        numeric_indices (List[int]): количественные признаки
        cat_indices (List[int]): категориальные признаки
    """
    transformers = []

    if numeric_indices:
        transformers.append(
            (
                "num",
                StandardScaler(),
                numeric_indices,
            )
        )

    if cat_indices:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                cat_indices,
            )
        )

    # Если есть что трансформировать — используем ColumnTransformer.
    # Если вдруг нет ни одного признака (крайний случай) — можно обойтись без него.
    if transformers:
        preprocessor = ColumnTransformer(transformers=transformers)
        cls = models_registry.resolve_model_class("logistic_regression")
        logreg = cls(**params)
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("clf", logreg),
            ]
        )
    else:
        # На всякий случай fallback: просто LogisticRegression без препроцессинга
        cls = models_registry.resolve_model_class("logistic_regression")
        model = cls(**params)

    return model


def list_available_model_classes() -> List[Dict[str, Any]]:
    """возвращает список допустимых классов моделей

    Returns:
        List[Dict[str, Any]]: словарь со значениями
            - key
            - display_name
            - hyperparams (описание гиперпараметров)
    """
    return models_registry.list_available_model_classes()


def list_trained_models() -> List[ModelMetadata]:
    """возвращает список всех моделей (включая помеченные как 'deleted')

    Returns:
        List[ModelMetadata]: список всех моделей
    """
    return storage.list_models()


def get_model_info(model_id: str) -> ModelMetadata:
    """возвращает метаданные конкретной модели по её id

    Args:
        model_id (str): id модели

    Returns:
        ModelMetadata: метаданные конкретной модели
    """
    return storage.get_model_metadata(model_id)


def train_model(
    model_class_key: str,
    X: Sequence[Sequence[Any]],
    y: Sequence[Any],
    hyperparams: Optional[Dict[str, Any]] = None,
    feature_types: Optional[Sequence[Optional[str]]] = None,
) -> ModelMetadata:
    """обучает новую модель указанного класса и сохраняет ее в хранилище

    Args:
        model_class_key (str): класс модели
        X (Sequence[Sequence[float]]): признаки
        y (Sequence[Any]): таргет
        hyperparams (Optional[Dict[str, Any]], optional): гиперпараметры. Defaults to None.
        feature_types (Optional[Sequence[Optional[str]]], optional: список типов признаков ('num'/'cat') длиной n_features
            Если None — типы признаков определяются автоматически по типам значений в X

    Returns:
        ModelMetadata: метаданные созданной модели
    """
    n_features = _validate_X_and_y(X, y)
    logger.info(
        "Запуск обучения новой модели: class=%s, n_samples=%d, n_features=%d",
        model_class_key,
        len(X),
        n_features,
    )
    
    task = _start_clearml_task(
        action="train",
        model_class_key=model_class_key,
        extra_config={
            "hyperparams": hyperparams or {},
            "feature_types": (
                list(feature_types) if feature_types is not None else None
            ),
            "n_samples": len(X),
            "n_features": n_features,
        },
    )

    params = _validate_and_merge_hyperparams(model_class_key, hyperparams)

    numeric_indices, cat_indices = _infer_feature_indices(X, feature_types)
    logger.info(
        "Типы признаков: numeric=%s, categorical=%s",
        numeric_indices,
        cat_indices,
    )

    if model_class_key == "logistic_regression":
        # Строим pipeline с препроцессингом
        model = _build_logistic_pipeline(params, numeric_indices, cat_indices)
        model.fit(X, y)

    elif model_class_key == "catboost_classifier":
        # Для CatBoost — обычный CatBoostClassifier, cat_features передаём в fit
        model_cls = models_registry.resolve_model_class(model_class_key)
        model = model_cls(**params)
        if cat_indices:
            model.fit(X, y, cat_features=cat_indices)
        else:
            model.fit(X, y)
    else:
        model_cls = models_registry.resolve_model_class(model_class_key)
        model = model_cls(**params)
        model.fit(X, y)

    metrics: Dict[str, Any] = {}
    if hasattr(model, "score"):
        try:
            score = model.score(X, y)
            metrics["train_score"] = float(score)
        except Exception:  # noqa: BLE001
            logger.warning(
                "Не удалось посчитать train_score для модели %s", model_class_key
            )

    meta = storage.save_new_model(
        model_class_key=model_class_key,
        hyperparams=params,
        model_obj=model,
        metrics=metrics,
        status="trained",
    )

    logger.info(
        "Обучение модели завершено: id=%s, class=%s, metrics=%s",
        meta.id,
        meta.model_class_key,
        meta.metrics,
    )
    
    if task is not None:
        try:
            cm_logger = task.get_logger()
            for name, value in metrics.items():
                try:
                    cm_logger.report_scalar(
                        title="train_metrics",
                        series=name,
                        value=float(value),
                        iteration=0,
                    )
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "Не удалось отправить метрику %s в ClearML (train)", name
                    )
            try:
                # просто удобно, чтобы в ClearML было видно id модели
                task.set_comment(f"model_id={meta.id}")
            except Exception:
                pass
        finally:
            task.close()
            
    return meta


def predict(
    model_id: str,
    X: Sequence[Sequence[Any]],
) -> Dict[str, Any]:
    """считает предсказания по уже обученной модели

    Args:
        model_id (str): id модели
        X (Sequence[Sequence[Any]]): признаки

    Returns:
        Dict[str, Any]: словарь с предсказаниями {'model_id': <>, 'predictions': <>, 'probabilities': <>}
    """
    logger.info("Запрос на предсказание: model_id=%s, n_samples=%d", model_id, len(X))

    model = storage.load_model(model_id)

    raw_preds = model.predict(X)
    predictions = [_to_primitive(p) for p in raw_preds]

    result: Dict[str, Any] = {
        "model_id": model_id,
        "predictions": predictions,
    }

    if hasattr(model, "predict_proba"):
        try:
            raw_probas = model.predict_proba(X)
            result["probabilities"] = [
                [_to_primitive(p) for p in row] for row in raw_probas
            ]
        except Exception:
            logger.warning(
                "Не удалось получить predict_proba для model_id=%s", model_id
            )

    logger.info("Предсказание успешно выполнено: model_id=%s", model_id)
    return result


def retrain_model(
    model_id: str,
    X: Sequence[Sequence[Any]],
    y: Sequence[Any],
    hyperparams: Optional[Dict[str, Any]] = None,
    feature_types: Optional[Sequence[Optional[str]]] = None,
) -> ModelMetadata:
    """переобучает уже созданную модель

    Упрощённый вариант:
        - берём исходный класс модели (из метаданных)
        - валидируем/сливаем гиперпараметры (если hyperparams=None — берём старые)
        - определяем типы признаков (num/cat) — так же, как в train_model
        - создаём НОВЫЙ экземпляр модели и обучаем его с нуля
        - перезаписываем .pkl и метаданные (updated_at, hyperparams, metrics)
    То есть фактически "обучение заново" под тем же id.

    Args:
        model_id (str): id модели
        X (Sequence[Sequence[Any]]): признаки
        y (Sequence[Any]): таргет
        hyperparams (Optional[Dict[str, Any]], optional): гиперпараметры модели. Defaults to None.
        feature_types (Optional[Sequence[Optional[str]]], optional): типы признаков. Defaults to None.

    Returns:
        ModelMetadata: метаданные модели
    """
    n_features = _validate_X_and_y(X, y)
    old_meta = storage.get_model_metadata(model_id)
    model_class_key = old_meta.model_class_key

    logger.info(
        "Запуск переобучения модели: id=%s, class=%s, n_samples=%d, n_features=%d",
        model_id,
        model_class_key,
        len(X),
        n_features,
    )
    
    task = _start_clearml_task(
        action="retrain",
        model_class_key=model_class_key,
        extra_config={
            "model_id": model_id,
            "old_hyperparams": old_meta.hyperparams,
            "new_hyperparams": hyperparams or old_meta.hyperparams,
            "n_samples": len(X),
            "n_features": n_features,
        },
    )

    # Если новые гиперпараметры не заданы, используем старые
    if hyperparams is None:
        params = dict(old_meta.hyperparams)
    else:
        params = _validate_and_merge_hyperparams(model_class_key, hyperparams)

    numeric_indices, cat_indices = _infer_feature_indices(X, feature_types)
    logger.info(
        "Типы признаков (retrain): numeric=%s, categorical=%s",
        numeric_indices,
        cat_indices,
    )

    # Строим/обучаем модель аналогично train_model
    if model_class_key == "logistic_regression":
        model = _build_logistic_pipeline(params, numeric_indices, cat_indices)
        model.fit(X, y)
    elif model_class_key == "catboost_classifier":
        model_cls = models_registry.resolve_model_class(model_class_key)
        model = model_cls(**params)
        if cat_indices:
            model.fit(X, y, cat_features=cat_indices)
        else:
            model.fit(X, y)
    else:
        model_cls = models_registry.resolve_model_class(model_class_key)
        model = model_cls(**params)
        model.fit(X, y)

    metrics: Dict[str, Any] = {}
    if hasattr(model, "score"):
        try:
            score = model.score(X, y)
            metrics["train_score"] = float(score)
        except Exception:  # noqa: BLE001
            logger.warning(
                "Не удалось посчитать train_score при переобучении model_id=%s",
                model_id,
            )

    # Обновляем модель в storage
    new_meta = storage.update_existing_model(
        model_id=model_id,
        model_obj=model,
        hyperparams=params,
        metrics=metrics,
        status="trained",
    )

    logger.info(
        "Переобучение модели завершено: id=%s, class=%s, metrics=%s",
        new_meta.id,
        new_meta.model_class_key,
        new_meta.metrics,
    )
    
    if task is not None:
        try:
            cm_logger = task.get_logger()
            for name, value in metrics.items():
                try:
                    cm_logger.report_scalar(
                        title="retrain_metrics",
                        series=name,
                        value=float(value),
                        iteration=0,
                    )
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "Не удалось отправить метрику %s в ClearML (retrain)", name
                    )
            try:
                task.set_comment(f"model_id={new_meta.id}")
            except Exception:
                pass
        finally:
            task.close()
            
    return new_meta


def delete_model(model_id: str, hard_delete: bool = False) -> None:
    """удаляет модель с указанным id

    Args:
        model_id (str): id модели для удаления
        hard_delete (bool, optional): если True => запись метаданных полностью удаляется из json. Defaults to False.
    """
    logger.info("Удаление модели: id=%s, hard_delete=%s", model_id, hard_delete)
    storage.delete_model(model_id, hard_delete=hard_delete)
    logger.info("Модель %s успешно удалена.", model_id)
