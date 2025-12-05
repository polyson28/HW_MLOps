from app.schemas import ModelClassInfo

def get_catboost_classifier_info() -> ModelClassInfo:
    return ModelClassInfo(
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
    )
