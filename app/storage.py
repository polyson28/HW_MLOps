# хранилище обученных моделей и мета данных о них

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
from pathlib import Path
import json
import logging
import threading
from uuid import uuid4

import joblib

logger = logging.getLogger("ml_service")


@dataclass
class ModelMetadata:
    """описание сохраненной модели
    id: уникальный идентификатор модели
    model_class_key: ключ класса модели
    hyperparams: гиперпараметры, с которыми модель обучена
    status: статус модели: "trained" / "deleted" / ...
    created_at: ISO-строка с датой/временем создания
    updated_at: ISO-строка с датой/временем последнего обновления
    model_path: путь к файлу с моделью
    metrics: словарь с метриками
    """

    id: str
    model_class_key: str
    hyperparams: Dict[str, Any]
    status: str
    created_at: str
    updated_at: str
    model_path: str
    metrics: Dict[str, Any]


class ModelStorage:
    """класс - обертка над файловым хранилищем моделей:
    - self._db: словарь {model_id: dict(metadata)}
    - self._db_path: путь к JSON с метаданными
    - директория с .pkl файлами
    """

    def __init__(
        self,
        base_dir: str | Path = "models_storage",
        db_filename: str = "models_db.json",
    ) -> None:
        """инициализация параметров

        Args:
            base_dir (str | Path, optional): директория для сохранения модели / моделей. Defaults to "models_storage".
            db_filename (str, optional): имя файла для сохранения модели. Defaults to "models_db.json".
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self._db_path = self.base_dir / db_filename
        self._lock = threading.Lock()
        self._db: Dict[str, Dict[str, Any]] = {}

        self._load_db()

    def _load_db(self) -> None:
        """чтение json с метаданными, если он существует

        Raises:
            RuntimeError: при неверном формате файла метаданных
            RuntimeError: при ошибке чтения файла
        """
        if not self._db_path.exists():
            logger.info(
                "Файл метаданных %s не найден, создаём пустую БД.", self._db_path
            )
            self._db = {}
            return

        try:
            with self._db_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                # Ожидаем словарь {model_id -> metadata_dict}
                if not isinstance(data, dict):
                    raise RuntimeError(
                        "Файл метаданных имеет неверный формат (ожидается dict)."
                    )
                self._db = data
        except (OSError, json.JSONDecodeError) as exc:
            logger.exception("Ошибка при чтении файла метаданных %s", self._db_path)
            raise RuntimeError(
                f"Не удалось прочитать файл метаданных: {self._db_path}"
            ) from exc

    def _save_db(self) -> None:
        """Сcхраняет текущий self._db в json

        Raises:
            RuntimeError: если не удалось сохранить файл метаданных
        """
        try:
            with self._db_path.open("w", encoding="utf-8") as f:
                json.dump(self._db, f, ensure_ascii=False, indent=2)
        except OSError as exc:
            logger.exception("Ошибка при записи файла метаданных %s", self._db_path)
            raise RuntimeError(
                f"Не удалось сохранить файл метаданных: {self._db_path}"
            ) from exc

    def list_models(self) -> List[ModelMetadata]:
        """возвращает список всех моделей (включая потенциально удалённые)"""
        with self._lock:
            return [ModelMetadata(**meta) for meta in self._db.values()]

    def get_model_metadata(self, model_id: str) -> ModelMetadata:
        """возвращает метаданные модели по её id

        Raises:
            KeyError: если модели с таким id нет
        """
        with self._lock:
            meta_dict = self._db.get(model_id)
            if meta_dict is None:
                raise KeyError(f"Модель с id={model_id!r} не найдена.")
            return ModelMetadata(**meta_dict)

    def save_new_model(
        self,
        model_class_key: str,
        hyperparams: Dict[str, Any],
        model_obj: Any,
        metrics: Optional[Dict[str, Any]] = None,
        status: str = "trained",
    ) -> ModelMetadata:
        """создание нового model_id, сохранение объекта модели и метаданных

        Args:
            model_class_key (str): название класса модели
            hyperparams (Dict[str, Any]): словарь с гиперпараметрами модели
            model_obj (Any): модель
            metrics (Optional[Dict[str, Any]], optional): метрики. Defaults to None.
            status (str, optional): статус модели. Defaults to "trained".

        Raises:
            RuntimeError: если не удалось сериализовать модель или сохранить метаданные

        Returns:
            ModelMetadata: созданный ModelMetadata
        """
        from datetime import datetime

        model_id = str(uuid4())
        model_path = self.base_dir / f"{model_id}.pkl"

        # Сериализуем саму модель
        try:
            joblib.dump(model_obj, model_path)
        except Exception as exc:
            logger.exception("Не удалось сохранить модель в файл %s", model_path)
            raise RuntimeError(
                f"Не удалось сохранить модель в файл: {model_path}"
            ) from exc

        now = datetime.utcnow().isoformat() + "Z"
        meta = ModelMetadata(
            id=model_id,
            model_class_key=model_class_key,
            hyperparams=hyperparams,
            status=status,
            created_at=now,
            updated_at=now,
            model_path=str(model_path),
            metrics=metrics or {},
        )

        with self._lock:
            self._db[model_id] = asdict(meta)
            self._save_db()

        logger.info(
            "Сохранена новая модель (id=%s, class=%s) в файл %s",
            model_id,
            model_class_key,
            model_path,
        )
        return meta

    def update_existing_model(
        self,
        model_id: str,
        model_obj: Any,
        hyperparams: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        status: Optional[str] = None,
    ) -> ModelMetadata:
        """обновление уже существующей модели (перезаписывание файла pkl и обновление метаданных)

        Args:
            model_id (str): id модели
            model_obj (Any): модель
            hyperparams (Optional[Dict[str, Any]], optional): гиперпараметры модели. Defaults to None.
            metrics (Optional[Dict[str, Any]], optional): метрики. Defaults to None.
            status (Optional[str], optional): статус модели. Defaults to None.

        Raises:
            KeyError: если модель не найден
            RuntimeError: если не удалось обновить модель или сохранить метаданные

        Returns:
            ModelMetadata: обновленный ModelMetadata
        """
        from datetime import datetime

        with self._lock:
            meta_dict = self._db.get(model_id)
            if meta_dict is None:
                raise KeyError(f"Модель с id={model_id!r} не найдена.")

            meta = ModelMetadata(**meta_dict)
            model_path = Path(meta.model_path)

            # Обновляем файл модели
            try:
                joblib.dump(model_obj, model_path)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Не удалось обновить модель в файле %s", model_path)
                raise RuntimeError(
                    f"Не удалось обновить модель в файле: {model_path}"
                ) from exc

            # Обновляем метаданные
            if hyperparams is not None:
                meta.hyperparams = hyperparams
            if metrics is not None:
                meta.metrics = metrics
            if status is not None:
                meta.status = status

            meta.updated_at = datetime.utcnow().isoformat() + "Z"

            self._db[model_id] = asdict(meta)
            self._save_db()

        logger.info(
            "Обновлена модель (id=%s). Статус=%s",
            model_id,
            meta.status,
        )
        return meta

    def load_model(self, model_id: str) -> Any:
        """загрузка модели по id

        Args:
            model_id (str): id модели

        Raises:
            FileNotFoundError: если файл с моделью отсутствует
            RuntimeError: если не удалось десериализовать модель
        """
        meta = self.get_model_metadata(model_id)
        model_path = Path(meta.model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")

        try:
            model_obj = joblib.load(model_path)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Не удалось загрузить модель из файла %s", model_path)
            raise RuntimeError(
                f"Не удалось загрузить модель из файла: {model_path}"
            ) from exc

        return model_obj

    def delete_model(self, model_id: str, hard_delete: bool = False) -> None:
        """удаление модели

        Args:
            model_id (str): id модели
            hard_delete (bool, optional): режим удаления, если True => полностью убираем запись из self._db,
                иначе в метаданных статус выставляем 'deleted'. Defaults to False.

        Raises:
            KeyError: если модель не найдена
        """
        with self._lock:
            meta_dict = self._db.get(model_id)
            if meta_dict is None:
                raise KeyError(f"Модель с id={model_id!r} не найдена.")

            meta = ModelMetadata(**meta_dict)
            model_path = Path(meta.model_path)

            # Пытаемся удалить файл, если он есть
            if model_path.exists():
                try:
                    model_path.unlink()
                    logger.info("Удалён файл модели %s (id=%s)", model_path, model_id)
                except OSError as exc:
                    logger.warning(
                        "Не удалось удалить файл модели %s: %s", model_path, exc
                    )

            if hard_delete:
                # Полностью убираем запись из БД
                del self._db[model_id]
            else:
                # Помечаем модель как удалённую
                meta.status = "deleted"
                self._db[model_id] = asdict(meta)

            self._save_db()
            logger.info("Модель %s удалена (hard_delete=%s).", model_id, hard_delete)

