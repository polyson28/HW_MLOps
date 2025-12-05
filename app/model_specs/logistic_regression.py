from app.schemas import ModelClassInfo

def get_logistic_regression_info() -> ModelClassInfo:
    return ModelClassInfo(
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
    )