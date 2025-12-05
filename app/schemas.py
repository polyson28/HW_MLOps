from dataclasses import dataclass
from typing import Any, Dict

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