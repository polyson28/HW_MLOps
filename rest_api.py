from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import Any, Dict, List, Optional, Union
import logging
import uvicorn
import sys
from pathlib import Path
from dataclasses import asdict

sys.path.insert(0, str(Path(__file__).parent))

from app import ml_core
from app.models_registry import list_available_model_classes


# система логирования всего приложения
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ml_service")


# cоздание FastAPI приложения
app = FastAPI(
    title="ML-service API",
    description="API для обучения, управления и использования ML-моделей",
    version="1.0.0",
)


# Pydantic модели для валидации запросов и ответов
class TrainRequest(BaseModel):
    """Запрос на обучение новой модели"""

    model_class_key: str = Field(
        ..., description="Класс модели ('catboost' или 'logistic')"
    )
    X: List[List[Any]] = Field(..., description="Обучающие данные")
    y: List[Any] = Field(..., description="Таргет")
    hyperparams: Optional[Dict[str, Any]] = Field(
        default=None, description="Гиперпараметры модели"
    )
    feature_types: Optional[List[str]] = Field(
        default=None, description="Типы признаков: 'numeric' или 'categorical'"
    )

    @field_validator("X")  # специальный декоратор
    @classmethod  # используем, тк валидация происходит до создания объекта: при инициализации Pydantic ещё не создал объект, сначала вызывает валидатор validate_X_not_empty() (тут объекта ещё нет, есть только класс)
    def validate_X_not_empty(cls, v):
        if not v or not v[0]:
            raise ValueError("X не может быть пустым")
        return v

    @field_validator("y")
    @classmethod
    def validate_y_not_empty(cls, v):
        if not v:
            raise ValueError("y не может быть пустым")
        return v


class TrainResponse(BaseModel):
    """Ответ после успешного обучения"""

    model_id: str = Field(..., description="Уникальный идентификатор обученной модели")
    model_class_key: str = Field(..., description="Класс модели")
    hyperparams: Dict[str, Any] = Field(
        ..., description="Использованные гиперпараметры"
    )
    metrics: Dict[str, Any] = Field(..., description="Метрики обучения")
    message: str = Field(default="Модель успешно обучена")


class PredictRequest(BaseModel):
    """Запрос на получение предсказаний"""

    model_id: str = Field(..., description="ID модели для предсказания")
    X: List[List[Any]] = Field(..., description="Данные для предсказания")

    @field_validator("X")
    @classmethod
    def validate_X_not_empty(cls, v):
        if not v:
            raise ValueError("X не может быть пустым")
        return v


class PredictResponse(BaseModel):
    """Ответ с предсказаниями"""

    model_id: str
    predictions: List[Any] = Field(..., description="Предсказания модели")
    probabilities: Optional[List[List[float]]] = Field(
        default=None, description="Вероятности классов (если доступны)"
    )


class RetrainRequest(BaseModel):
    """Запрос на переобучение существующей модели"""

    model_id: str = Field(..., description="ID модели для переобучения")
    X: List[List[Any]] = Field(..., description="Новые обучающие данные")
    y: List[Any] = Field(..., description="Новая целевая переменная")
    hyperparams: Optional[Dict[str, Any]] = Field(
        default=None, description="Новые гиперпараметры (опционально)"
    )
    feature_types: Optional[List[str]] = Field(
        default=None, description="Типы признаков"
    )

    @field_validator("X")
    @classmethod
    def validate_X_not_empty(cls, v):
        if not v or not v[0]:
            raise ValueError("X не может быть пустым")
        return v

    @field_validator("y")
    @classmethod
    def validate_y_not_empty(cls, v):
        if not v:
            raise ValueError("y не может быть пустым")
        return v


class RetrainResponse(BaseModel):
    """Ответ после переобучения"""

    model_id: str
    model_class_key: str
    hyperparams: Dict[str, Any]
    metrics: Dict[str, Any]
    message: str = Field(default="Модель успешно переобучена")


class DeleteResponse(BaseModel):
    """Ответ после удаления модели"""

    model_id: str
    message: str


class ModelInfo(BaseModel):
    """Информация о модели"""

    id: str
    model_class_key: str
    hyperparams: Dict[str, Any]
    status: str
    created_at: str
    updated_at: str
    metrics: Dict[str, Any]


class ModelClassInfo(BaseModel):
    """Информация о доступном классе модели"""

    key: str
    display_name: str
    param_schema: Dict[str, Dict[str, Any]]


class HealthResponse(BaseModel):
    """Ответ статуса сервиса"""

    status: str
    message: str


# Эндпоинты API - URL-адрес, по которому клиентское приложение может обратиться к серверу для выполнения определённого действия / получения данных
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Проверка статуса сервиса
    Возвращает статус работы API
    """
    return HealthResponse(status="ok", message="ML Service is running")


@app.get("/models/available", response_model=List[ModelClassInfo], tags=["Models"])
async def get_available_models():
    """
    Получить список доступных классов моделей для обучения
    Возвращает информацию о всех поддерживаемых моделях, их параметрах и схемах валидации
    """
    try:
        models_info = list_available_model_classes()
        result = []
        for info in models_info:
            model_info = ModelClassInfo(
                key=info["key"],
                display_name=info["display_name"],
                param_schema=info["hyperparams"],
            )
            result.append(model_info)

        return result
    except KeyError as e:
        logger.error(
            f"Ошибка при получении списка доступных моделей: отсутствует ключ {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка в конфигурации моделей: отсутствует поле {e}",
        )
    except Exception as e:
        logger.error(f"Ошибка при получении списка доступных моделей: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Внутренняя ошибка сервера: {str(e)}",
        )


@app.post(
    "/train",
    response_model=TrainResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Training"],
)
async def train_model(request: TrainRequest):
    """
    Обучить новую модель
    Принимает данные, класс модели и гиперпараметры, обучает модель и сохраняет её.
    Возвращает ID обученной модели и метрики.
    """
    try:
        logger.info(f"Начало обучения модели класса '{request.model_class_key}'")

        result = ml_core.train_model(
            model_class_key=request.model_class_key,
            X=request.X,
            y=request.y,
            hyperparams=request.hyperparams,
            feature_types=request.feature_types,
        )

        logger.info(f"Модель '{result.id}' успешно обучена")

        return TrainResponse(
            model_id=result.id,
            model_class_key=result.model_class_key,
            hyperparams=result.hyperparams,
            metrics=result.metrics,
        )

    except ValueError as e:
        logger.warning(f"Ошибка валидации при обучении: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Ошибка при обучении модели: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Внутренняя ошибка при обучении модели: {str(e)}",
        )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(request: PredictRequest):
    """
    Получить предсказания модели
    Загружает обученную модель по ID и возвращает предсказания для переданных данных.
    Если модель поддерживает вероятности классов, возвращает их тоже.
    """
    try:
        logger.info(f"Запрос предсказания для модели '{request.model_id}'")

        result = ml_core.predict(model_id=request.model_id, X=request.X)

        return PredictResponse(
            model_id=request.model_id,
            predictions=result["predictions"],
            probabilities=result.get("probabilities"),
        )

    except FileNotFoundError as e:
        logger.warning(f"Модель не найдена: {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        logger.warning(f"Ошибка валидации при предсказании: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Ошибка при предсказании: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Внутренняя ошибка при предсказании: {str(e)}",
        )


@app.post("/retrain", response_model=RetrainResponse, tags=["Training"])
async def retrain_model(request: RetrainRequest):
    """
    Переобучить существующую модель
    Загружает модель, переобучает её на новых данных (опционально с новыми гиперпараметрами)
    и сохраняет обновлённую версию.
    """
    try:
        logger.info(f"Начало переобучения модели '{request.model_id}'")

        result = ml_core.retrain_model(
            model_id=request.model_id,
            X=request.X,
            y=request.y,
            hyperparams=request.hyperparams,
            feature_types=request.feature_types,
        )

        logger.info(f"Модель '{request.model_id}' успешно переобучена")

        return RetrainResponse(
            model_id=result.id,
            model_class_key=result.model_class_key,
            hyperparams=result.hyperparams,
            metrics=result.metrics,
        )

    except FileNotFoundError as e:
        logger.warning(f"Модель не найдена: {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        logger.warning(f"Ошибка валидации при переобучении: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Ошибка при переобучении модели: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Внутренняя ошибка при переобучении: {str(e)}",
        )


@app.delete("/models/{model_id}", response_model=DeleteResponse, tags=["Models"])
async def delete_model(model_id: str, hard: bool = False):
    """
    Удалить модель
    Параметры:
    - model_id: ID модели для удаления
    - hard: если True, удаляет файлы модели с диска; если False (по умолчанию), только помечает как удалённую
    """
    try:
        logger.info(f"Удаление модели '{model_id}' (hard={hard})")

        ml_core.delete_model(model_id=model_id, hard_delete=hard)

        delete_type = "полностью удалена" if hard else "помечена как удалённая"
        logger.info(f"Модель '{model_id}' {delete_type}")

        return DeleteResponse(model_id=model_id, message=f"Модель {delete_type}")

    except FileNotFoundError as e:
        logger.warning(f"Модель не найдена: {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Ошибка при удалении модели: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Внутренняя ошибка при удалении: {str(e)}",
        )


@app.get("/models", response_model=List[ModelInfo], tags=["Models"])
async def list_models():
    """
    Получить список всех обученных моделей
    Возвращает метаданные всех сохранённых моделей
    """
    try:
        models = ml_core.list_trained_models()
        return [
            ModelInfo(
                id=m.id,
                model_class_key=m.model_class_key,
                hyperparams=m.hyperparams,
                status=m.status,
                created_at=m.created_at,
                updated_at=m.updated_at,
                metrics=m.metrics,
            )
            for m in models
        ]
    except Exception as e:
        logger.error(f"Ошибка при получении списка моделей: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Внутренняя ошибка при получении списка моделей: {str(e)}",
        )


@app.get("/models/{model_id}", response_model=ModelInfo, tags=["Models"])
async def get_model_info(model_id: str):
    """
    Получить информацию о конкретной модели
    Возвращает метаданные модели по её ID
    """
    try:
        info = ml_core.get_model_info(model_id=model_id)
        return ModelInfo(
            id=info.id,
            model_class_key=info.model_class_key,
            hyperparams=info.hyperparams,
            status=info.status,
            created_at=info.created_at,
            updated_at=info.updated_at,
            metrics=info.metrics,
        )
    except FileNotFoundError as e:
        logger.warning(f"Модель не найдена: {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Ошибка при получении информации о модели: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Внутренняя ошибка: {str(e)}",
        )


# Запуск сервера
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
