import grpc
from concurrent import futures
import logging
import sys
import json
from pathlib import Path

# Добавляем корневую директорию и app/ в путь для импорта
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import ml_core
from app.models_registry import list_available_model_classes

# Импорт сгенерированных protobuf файлов
import grpc_service.ml_service_pb2 as ml_pb2
import grpc_service.ml_service_pb2_grpc as ml_pb2_grpc

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ml_service_grpc")


def _proto_value_to_python(value):
    """Конвертирует protobuf Value в Python значение"""
    if value.HasField("number"):
        return value.number
    elif value.HasField("text"):
        return value.text
    return None


def _python_to_proto_value(val):
    """Конвертирует Python значение в protobuf Value"""
    value = ml_pb2.Value()
    if isinstance(val, (int, float)):
        value.number = float(val)
    else:
        value.text = str(val)
    return value


def _proto_to_python_matrix(proto_rows):
    """Конвертирует repeated DataRow в Python list[list]"""
    return [
        [_proto_value_to_python(val) for val in row.values]
        for row in proto_rows
    ]


def _proto_to_python_vector(proto_values):
    """Конвертирует repeated Value в Python list"""
    return [_proto_value_to_python(val) for val in proto_values]


def _string_dict_to_typed_dict(str_dict):
    """Конвертирует словарь строк в словарь с типизированными значениями"""
    result = {}
    for key, val in str_dict.items():
        # Пытаемся преобразовать в число
        try:
            if '.' in val:
                result[key] = float(val)
            else:
                result[key] = int(val)
        except (ValueError, AttributeError):
            # Если не число, оставляем строкой
            result[key] = val
    return result


class MLServiceServicer(ml_pb2_grpc.MLServiceServicer):
    """Реализация gRPC сервиса для ML операций"""

    def Train(self, request, context):
        """Обучение новой модели"""
        try:
            logger.info(f"gRPC Train: model_class={request.model_class}")
            
            # Конвертация данных
            X = _proto_to_python_matrix(request.X)
            y = _proto_to_python_vector(request.y)
            hyperparams = _string_dict_to_typed_dict(dict(request.hyperparams)) if request.hyperparams else None
            feature_types = list(request.feature_types) if request.feature_types else None
            
            # Обучение модели
            result = ml_core.train_model(
                model_class=request.model_class,
                X=X,
                y=y,
                hyperparams=hyperparams,
                feature_types=feature_types
            )
            
            # Формирование ответа
            response = ml_pb2.TrainResponse(
                model_id=result["model_id"],
                model_class=result["model_class"],
                hyperparams={k: str(v) for k, v in result["hyperparams"].items()},
                metrics={k: float(v) for k, v in result["metrics"].items()},
                message="Модель успешно обучена"
            )
            
            logger.info(f"gRPC Train: успешно обучена модель {result['model_id']}")
            return response
            
        except ValueError as e:
            logger.warning(f"gRPC Train: ошибка валидации - {e}")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return ml_pb2.TrainResponse()
            
        except Exception as e:
            logger.error(f"gRPC Train: внутренняя ошибка - {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Внутренняя ошибка: {str(e)}")
            return ml_pb2.TrainResponse()

    def ListAvailableModels(self, request, context):
        """Получение списка доступных классов моделей"""
        try:
            logger.info("gRPC ListAvailableModels")
            
            models_info = list_available_model_classes()
            
            response = ml_pb2.AvailableModelsResponse()
            for info in models_info:
                model_info = ml_pb2.ModelClassInfo(
                    key=info["key"],
                    display_name=info["display_name"],
                    default_params={k: str(v) for k, v in info["default_params"].items()},
                    param_schema_json=json.dumps(info["param_schema"], ensure_ascii=False)
                )
                response.models.append(model_info)
            
            return response
            
        except Exception as e:
            logger.error(f"gRPC ListAvailableModels: ошибка - {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return ml_pb2.AvailableModelsResponse()

    def Predict(self, request, context):
        """Получение предсказаний"""
        try:
            logger.info(f"gRPC Predict: model_id={request.model_id}")
            
            X = _proto_to_python_matrix(request.X)
            
            result = ml_core.predict(
                model_id=request.model_id,
                X=X
            )
            
            # Формирование ответа
            response = ml_pb2.PredictResponse(
                model_id=request.model_id,
                predictions=[_python_to_proto_value(p) for p in result["predictions"]]
            )
            
            # Добавление вероятностей, если есть
            if result.get("probabilities"):
                for prob_row in result["probabilities"]:
                    prob_msg = ml_pb2.ProbabilityRow(probs=prob_row)
                    response.probabilities.append(prob_msg)
            
            logger.info(f"gRPC Predict: успешно получено {len(result['predictions'])} предсказаний")
            return response
            
        except FileNotFoundError as e:
            logger.warning(f"gRPC Predict: модель не найдена - {e}")
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(e))
            return ml_pb2.PredictResponse()
            
        except ValueError as e:
            logger.warning(f"gRPC Predict: ошибка валидации - {e}")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return ml_pb2.PredictResponse()
            
        except Exception as e:
            logger.error(f"gRPC Predict: внутренняя ошибка - {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return ml_pb2.PredictResponse()

    def Retrain(self, request, context):
        """Переобучение существующей модели"""
        try:
            logger.info(f"gRPC Retrain: model_id={request.model_id}")
            
            X = _proto_to_python_matrix(request.X)
            y = _proto_to_python_vector(request.y)
            hyperparams = _string_dict_to_typed_dict(dict(request.hyperparams)) if request.hyperparams else None
            feature_types = list(request.feature_types) if request.feature_types else None
            
            result = ml_core.retrain_model(
                model_id=request.model_id,
                X=X,
                y=y,
                hyperparams=hyperparams,
                feature_types=feature_types
            )
            
            response = ml_pb2.RetrainResponse(
                model_id=result["model_id"],
                model_class=result["model_class"],
                hyperparams={k: str(v) for k, v in result["hyperparams"].items()},
                metrics={k: float(v) for k, v in result["metrics"].items()},
                message="Модель успешно переобучена"
            )
            
            logger.info(f"gRPC Retrain: успешно переобучена модель {request.model_id}")
            return response
            
        except FileNotFoundError as e:
            logger.warning(f"gRPC Retrain: модель не найдена - {e}")
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(e))
            return ml_pb2.RetrainResponse()
            
        except ValueError as e:
            logger.warning(f"gRPC Retrain: ошибка валидации - {e}")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return ml_pb2.RetrainResponse()
            
        except Exception as e:
            logger.error(f"gRPC Retrain: внутренняя ошибка - {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return ml_pb2.RetrainResponse()

    def DeleteModel(self, request, context):
        """Удаление модели"""
        try:
            logger.info(f"gRPC DeleteModel: model_id={request.model_id}, hard={request.hard}")
            
            ml_core.delete_model(
                model_id=request.model_id,
                hard=request.hard
            )
            
            delete_type = "полностью удалена" if request.hard else "помечена как удалённая"
            response = ml_pb2.DeleteResponse(
                model_id=request.model_id,
                message=f"Модель {delete_type}"
            )
            
            logger.info(f"gRPC DeleteModel: модель {request.model_id} {delete_type}")
            return response
            
        except FileNotFoundError as e:
            logger.warning(f"gRPC DeleteModel: модель не найдена - {e}")
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(e))
            return ml_pb2.DeleteResponse()
            
        except Exception as e:
            logger.error(f"gRPC DeleteModel: внутренняя ошибка - {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return ml_pb2.DeleteResponse()

    def ListTrainedModels(self, request, context):
        """Получение списка обученных моделей"""
        try:
            logger.info("gRPC ListTrainedModels")
            
            models = ml_core.list_trained_models()
            
            response = ml_pb2.TrainedModelsResponse()
            for m in models:
                model_info = ml_pb2.ModelInfo(
                    id=m["id"],
                    model_class=m["model_class"],
                    hyperparams={k: str(v) for k, v in m["hyperparams"].items()},
                    status=m["status"],
                    created_at=m["created_at"],
                    updated_at=m["updated_at"],
                    metrics={k: float(v) for k, v in m["metrics"].items()}
                )
                response.models.append(model_info)
            
            return response
            
        except Exception as e:
            logger.error(f"gRPC ListTrainedModels: ошибка - {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return ml_pb2.TrainedModelsResponse()

    def GetModelInfo(self, request, context):
        """Получение информации о модели"""
        try:
            logger.info(f"gRPC GetModelInfo: model_id={request.model_id}")
            
            info = ml_core.get_model_info(model_id=request.model_id)
            
            model_info = ml_pb2.ModelInfo(
                id=info["id"],
                model_class=info["model_class"],
                hyperparams={k: str(v) for k, v in info["hyperparams"].items()},
                status=info["status"],
                created_at=info["created_at"],
                updated_at=info["updated_at"],
                metrics={k: float(v) for k, v in info["metrics"].items()}
            )
            
            response = ml_pb2.ModelInfoResponse(info=model_info)
            return response
            
        except FileNotFoundError as e:
            logger.warning(f"gRPC GetModelInfo: модель не найдена - {e}")
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(e))
            return ml_pb2.ModelInfoResponse()
            
        except Exception as e:
            logger.error(f"gRPC GetModelInfo: ошибка - {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return ml_pb2.ModelInfoResponse()

    def HealthCheck(self, request, context):
        """Проверка статуса сервиса"""
        logger.info("gRPC HealthCheck")
        return ml_pb2.HealthResponse(
            status="ok",
            message="gRPC ML Service is running"
        )


def serve():
    """Запуск gRPC сервера"""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    ml_pb2_grpc.add_MLServiceServicer_to_server(MLServiceServicer(), server)
    
    server_address = "[::]:50051"
    server.add_insecure_port(server_address)
    
    logger.info(f"gRPC сервер запущен на {server_address}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
