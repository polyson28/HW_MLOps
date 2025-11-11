import streamlit as st
import requests
import pandas as pd
import json
import os
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

# Конфигурация REST API
REST_API_URL = os.getenv("REST_API_URL", "http://localhost:8000")

# Настройка страницы
st.set_page_config(
    page_title="ML Service Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Заголовок приложения
st.title("🤖 ML Service Dashboard")
st.markdown("---")


# ============================================================================
# Вспомогательные функции для работы с API
# ============================================================================

def check_api_health() -> bool:
    """Проверка доступности API"""
    try:
        response = requests.get(f"{REST_API_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def get_available_models() -> List[Dict[str, Any]]:
    """Получить список доступных классов моделей"""
    try:
        response = requests.get(f"{REST_API_URL}/models/available")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Ошибка при получении списка моделей: {e}")
        return []


def get_trained_models() -> List[Dict[str, Any]]:
    """Получить список обученных моделей"""
    try:
        response = requests.get(f"{REST_API_URL}/models")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Ошибка при получении списка обученных моделей: {e}")
        return []


def train_model(model_class_key: str, X: List[List[Any]], y: List[Any], 
                hyperparams: Optional[Dict[str, Any]] = None,
                feature_types: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    """Обучить модель"""
    try:
        payload = {
            "model_class_key": model_class_key,
            "X": X,
            "y": y
        }
        if hyperparams:
            payload["hyperparams"] = hyperparams
        if feature_types:
            payload["feature_types"] = feature_types
            
        response = requests.post(f"{REST_API_URL}/train", json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Ошибка при обучении модели: {e}")
        return None


def predict(model_id: str, X: List[List[Any]]) -> Optional[Dict[str, Any]]:
    """Получить предсказания"""
    try:
        payload = {
            "model_id": model_id,
            "X": X
        }
        response = requests.post(f"{REST_API_URL}/predict", json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Ошибка при получении предсказаний: {e}")
        return None


def retrain_model(model_id: str, X: List[List[Any]], y: List[Any],
                  hyperparams: Optional[Dict[str, Any]] = None,
                  feature_types: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    """Переобучить модель"""
    try:
        payload = {
            "model_id": model_id,
            "X": X,
            "y": y
        }
        if hyperparams:
            payload["hyperparams"] = hyperparams
        if feature_types:
            payload["feature_types"] = feature_types
            
        response = requests.post(f"{REST_API_URL}/retrain", json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Ошибка при переобучении модели: {e}")
        return None


def delete_model(model_id: str, hard: bool = False) -> bool:
    """Удалить модель"""
    try:
        response = requests.delete(f"{REST_API_URL}/models/{model_id}?hard={hard}")
        response.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Ошибка при удалении модели: {e}")
        return False


def get_model_info(model_id: str) -> Optional[Dict[str, Any]]:
    """Получить информацию о модели"""
    try:
        response = requests.get(f"{REST_API_URL}/models/{model_id}")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Ошибка при получении информации о модели: {e}")
        return None


# ============================================================================
# Вспомогательные функции для работы с данными
# ============================================================================

def parse_csv_to_data(uploaded_file) -> Optional[pd.DataFrame]:
    """Загрузка CSV файла"""
    try:
        # Сначала попробуем стандартный разделитель (запятая)
        try:
            df = pd.read_csv(uploaded_file)
        except:
            # Если не получилось, сбросим указатель и попробуем точку с запятой
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=';')
        
        # Если первый столбец без имени, удалим его
        if df.columns[0] == '' or 'Unnamed' in str(df.columns[0]):
            df = df.iloc[:, 1:]
        
        return df
    except Exception as e:
        st.error(f"Ошибка при чтении CSV: {e}")
        return None

def dataframe_to_lists(df: pd.DataFrame, target_col: Optional[str] = None):
    """Конвертация DataFrame в формат для API с приведением к Python типам"""
    
    def convert_value(val):
        """Конвертирует numpy типы в нативные Python типы"""
        if pd.isna(val):
            return None
        if isinstance(val, (np.integer, np.floating)):
            return float(val)
        if isinstance(val, np.bool_):
            return bool(val)
        if isinstance(val, (int, float, str, bool)):
            return val
        return str(val)
    
    if target_col:
        X_raw = df.drop(columns=[target_col]).values.tolist()
        y_raw = df[target_col].values.tolist()
        
        # Рекурсивная конвертация всех значений
        X = [[convert_value(val) for val in row] for row in X_raw]
        y = [convert_value(val) for val in y_raw]
        
        return X, y
    else:
        X_raw = df.values.tolist()
        X = [[convert_value(val) for val in row] for row in X_raw]
        return X, None


def render_hyperparams_inputs(param_schema: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Отрисовка полей ввода для гиперпараметров"""
    hyperparams = {}
    
    for param_name, param_info in param_schema.items():
        param_type = param_info.get("type", "str")
        description = param_info.get("description", "")
        default = param_info.get("default")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if param_type == "int":
                min_val = param_info.get("min", 1)
                max_val = param_info.get("max", 1000)
                value = st.number_input(
                    f"{param_name}",
                    min_value=min_val,
                    max_value=max_val,
                    value=default if default is not None else min_val,
                    step=1,
                    help=description
                )
                hyperparams[param_name] = int(value)
                
            elif param_type == "float":
                min_val = param_info.get("min", 0.0)
                max_val = param_info.get("max", 1.0)
                value = st.number_input(
                    f"{param_name}",
                    min_value=min_val,
                    max_value=max_val,
                    value=default if default is not None else min_val,
                    format="%.4f",
                    help=description
                )
                hyperparams[param_name] = float(value)
                
            elif param_type == "str":
                allowed = param_info.get("allowed")
                if allowed:
                    value = st.selectbox(
                        f"{param_name}",
                        options=allowed,
                        index=allowed.index(default) if default in allowed else 0,
                        help=description
                    )
                else:
                    value = st.text_input(
                        f"{param_name}",
                        value=default if default else "",
                        help=description
                    )
                hyperparams[param_name] = value
                
            elif param_type == "bool":
                value = st.checkbox(
                    f"{param_name}",
                    value=default if default is not None else False,
                    help=description
                )
                hyperparams[param_name] = value
        
        with col2:
            st.caption(f"Тип: {param_type}")
    
    return hyperparams


# ============================================================================
# Проверка доступности API
# ============================================================================

with st.sidebar:
    st.header("⚙️ Настройки")
    
    # Проверка статуса API
    if check_api_health():
        st.success("✅ API доступен")
    else:
        st.error("❌ API недоступен")
        st.warning(f"Убедитесь, что REST API запущен на {REST_API_URL}")
        st.stop()
    
    st.markdown("---")
    
    # Навигация
    st.header("📋 Навигация")
    page = st.radio(
        "Выберите страницу:",
        [
            "🏠 Главная",
            "🎓 Обучение модели",
            "📊 Список моделей",
            "🔮 Предсказание",
            "🔄 Переобучение",
            "🗑️ Удаление модели"
        ]
    )


# ============================================================================
# Страницы дашборда
# ============================================================================

if page == "🏠 Главная":
    st.header("Добро пожаловать в ML Service Dashboard")
    
    st.markdown("""
    Этот дашборд позволяет управлять ML-моделями через интерактивный интерфейс.
    
    ### Доступные функции:
    
    - **🎓 Обучение модели**: Обучите новую модель на ваших данных
    - **📊 Список моделей**: Просмотрите все обученные модели и их метрики
    - **🔮 Предсказание**: Получите предсказания от обученной модели
    - **🔄 Переобучение**: Переобучите существующую модель на новых данных
    - **🗑️ Удаление модели**: Удалите ненужную модель
    
    ### Поддерживаемые модели:
    """)
    
    available_models = get_available_models()
    for model in available_models:
        with st.expander(f"**{model['display_name']}** (`{model['key']}`)"):
            st.write("**Доступные гиперпараметры:**")
            for param, schema in model['param_schema'].items():
                st.write(f"- `{param}` ({schema['type']}): {schema.get('description', 'Нет описания')}")


elif page == "🎓 Обучение модели":
    st.header("Обучение новой модели")
    
    # Выбор класса модели
    available_models = get_available_models()
    if not available_models:
        st.error("Не удалось загрузить список доступных моделей")
        st.stop()
    
    model_options = {m['display_name']: m for m in available_models}
    selected_model_name = st.selectbox(
        "Выберите класс модели",
        options=list(model_options.keys())
    )
    selected_model = model_options[selected_model_name]
    
    st.info(f"Выбрана модель: **{selected_model['display_name']}** (`{selected_model['key']}`)")
    
    # Настройка гиперпараметров
    st.subheader("Настройка гиперпараметров")
    use_custom_params = st.checkbox("Использовать пользовательские гиперпараметры")
    
    hyperparams = None
    if use_custom_params:
        hyperparams = render_hyperparams_inputs(selected_model['param_schema'])
    
    # Загрузка данных
    st.subheader("Загрузка данных")
    uploaded_file = st.file_uploader("Загрузите CSV файл с обучающими данными", type=["csv"])
    
    if uploaded_file:
        df = parse_csv_to_data(uploaded_file)
        if df is not None:
            st.write("**Предпросмотр данных:**")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Выбор целевой переменной
            target_col = st.selectbox("Выберите целевую переменную (target)", options=df.columns.tolist())
            
            # Опциональное указание типов признаков
            st.subheader("Типы признаков (опционально)")
            specify_types = st.checkbox("Указать типы признаков вручную")
            
            feature_types = None
            if specify_types:
                feature_cols = [col for col in df.columns if col != target_col]
                feature_types = []
                st.write("Укажите тип для каждого признака:")
                for col in feature_cols:
                    col_type = st.selectbox(
                        f"{col}",
                        options=["numeric", "categorical"],
                        key=f"type_{col}"
                    )
                    feature_types.append(col_type)
            
            # Кнопка обучения
            if st.button("🚀 Обучить модель", type="primary"):
                with st.spinner("Обучение модели..."):
                    X, y = dataframe_to_lists(df, target_col)
                    
                    result = train_model(
                        model_class_key=selected_model['key'],
                        X=X,
                        y=y,
                        hyperparams=hyperparams,
                        feature_types=feature_types
                    )
                    
                    if result:
                        st.success(f"✅ Модель успешно обучена! ID: `{result['model_id']}`")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Гиперпараметры:**")
                            st.json(result['hyperparams'])
                        
                        with col2:
                            st.write("**Метрики:**")
                            st.json(result['metrics'])


elif page == "📊 Список моделей":
    st.header("Список обученных моделей")
    
    # Кнопка обновления
    if st.button("🔄 Обновить список"):
        st.rerun()
    
    models = get_trained_models()
    
    if not models:
        st.info("Нет обученных моделей")
    else:
        st.write(f"**Всего моделей:** {len(models)}")
        
        # Таблица моделей
        models_df = pd.DataFrame([
            {
                "ID": m["id"][:8] + "...",
                "Класс": m["model_class_key"],
                "Статус": m["status"],
                "Создана": m["created_at"],
                "Обновлена": m["updated_at"]
            }
            for m in models
        ])
        
        st.dataframe(models_df, use_container_width=True)
        
        # Детальная информация о выбранной модели
        st.subheader("Детальная информация")
        selected_model_id = st.selectbox(
            "Выберите модель для просмотра деталей",
            options=[m["id"] for m in models],
            format_func=lambda x: f"{x[:8]}... ({next((m['model_class_key'] for m in models if m['id'] == x), '')})"
        )
        
        if selected_model_id:
            model_info = get_model_info(selected_model_id)
            if model_info:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Класс модели", model_info["model_class_key"])
                    st.metric("Статус", model_info["status"])
                
                with col2:
                    st.write("**Гиперпараметры:**")
                    st.json(model_info["hyperparams"])
                
                with col3:
                    st.write("**Метрики:**")
                    st.json(model_info["metrics"])
                
                st.write(f"**Создана:** {model_info['created_at']}")
                st.write(f"**Обновлена:** {model_info['updated_at']}")


elif page == "🔮 Предсказание":
    st.header("Получение предсказаний")
    
    # Выбор модели
    models = get_trained_models()
    
    if not models:
        st.warning("Нет доступных обученных моделей. Сначала обучите модель.")
        st.stop()
    
    trained_models = [m for m in models if m["status"] == "trained"]
    if not trained_models:
        st.warning("Нет моделей со статусом 'trained'")
        st.stop()
    
    selected_model_id = st.selectbox(
        "Выберите модель",
        options=[m["id"] for m in trained_models],
        format_func=lambda x: f"{x[:8]}... ({next((m['model_class_key'] for m in trained_models if m['id'] == x), '')})"
    )
    
    # Загрузка данных
    st.subheader("Загрузка данных для предсказания")
    uploaded_file = st.file_uploader("Загрузите CSV файл с признаками", type=["csv"])
    
    if uploaded_file:
        df = parse_csv_to_data(uploaded_file)
        if df is not None:
            st.write("**Предпросмотр данных:**")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Кнопка предсказания
            if st.button("🔮 Получить предсказания", type="primary"):
                with st.spinner("Получение предсказаний..."):
                    X, _ = dataframe_to_lists(df)
                    
                    result = predict(model_id=selected_model_id, X=X)
                    
                    if result:
                        st.success("✅ Предсказания получены!")
                        
                        # Отображение результатов
                        result_df = df.copy()
                        result_df["Prediction"] = result["predictions"]
                        
                        if result.get("probabilities"):
                            for i, probs in enumerate(result["probabilities"]):
                                for j, prob in enumerate(probs):
                                    result_df[f"Probability_Class_{j}"] = None
                            
                            for i, probs in enumerate(result["probabilities"]):
                                for j, prob in enumerate(probs):
                                    result_df.at[i, f"Probability_Class_{j}"] = prob
                        
                        st.write("**Результаты:**")
                        st.dataframe(result_df, use_container_width=True)
                        
                        # Скачивание результатов
                        csv = result_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Скачать результаты",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv"
                        )


elif page == "🔄 Переобучение":
    st.header("Переобучение существующей модели")
    
    # Выбор модели
    models = get_trained_models()
    
    if not models:
        st.warning("Нет доступных моделей для переобучения")
        st.stop()
    
    selected_model_id = st.selectbox(
        "Выберите модель для переобучения",
        options=[m["id"] for m in models],
        format_func=lambda x: f"{x[:8]}... ({next((m['model_class_key'] for m in models if m['id'] == x), '')})"
    )
    
    # Получение информации о модели
    model_info = get_model_info(selected_model_id)
    if model_info:
        st.info(f"**Текущая модель:** {model_info['model_class_key']}")
        st.write("**Текущие гиперпараметры:**")
        st.json(model_info['hyperparams'])
    
    # Настройка новых гиперпараметров
    st.subheader("Новые гиперпараметры (опционально)")
    change_hyperparams = st.checkbox("Изменить гиперпараметры")
    
    hyperparams = None
    if change_hyperparams and model_info:
        # Получаем схему параметров для этого класса модели
        available_models = get_available_models()
        model_schema = next((m for m in available_models if m['key'] == model_info['model_class_key']), None)
        
        if model_schema:
            hyperparams = render_hyperparams_inputs(model_schema['param_schema'])
    
    # Загрузка новых данных
    st.subheader("Загрузка новых обучающих данных")
    uploaded_file = st.file_uploader("Загрузите CSV файл", type=["csv"])
    
    if uploaded_file:
        df = parse_csv_to_data(uploaded_file)
        if df is not None:
            st.write("**Предпросмотр данных:**")
            st.dataframe(df.head(10), use_container_width=True)
            
            target_col = st.selectbox("Выберите целевую переменную", options=df.columns.tolist())
            
            # Опциональное указание типов признаков
            specify_types = st.checkbox("Указать типы признаков")
            feature_types = None
            
            if specify_types:
                feature_cols = [col for col in df.columns if col != target_col]
                feature_types = []
                for col in feature_cols:
                    col_type = st.selectbox(
                        f"{col}",
                        options=["numeric", "categorical"],
                        key=f"retrain_type_{col}"
                    )
                    feature_types.append(col_type)
            
            # Кнопка переобучения
            if st.button("🔄 Переобучить модель", type="primary"):
                with st.spinner("Переобучение модели..."):
                    X, y = dataframe_to_lists(df, target_col)
                    
                    result = retrain_model(
                        model_id=selected_model_id,
                        X=X,
                        y=y,
                        hyperparams=hyperparams,
                        feature_types=feature_types
                    )
                    
                    if result:
                        st.success(f"✅ Модель успешно переобучена!")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Новые гиперпараметры:**")
                            st.json(result['hyperparams'])
                        
                        with col2:
                            st.write("**Новые метрики:**")
                            st.json(result['metrics'])


elif page == "🗑️ Удаление модели":
    st.header("Удаление модели")
    
    models = get_trained_models()
    
    if not models:
        st.info("Нет моделей для удаления")
        st.stop()
    
    # Выбор модели
    selected_model_id = st.selectbox(
        "Выберите модель для удаления",
        options=[m["id"] for m in models],
        format_func=lambda x: f"{x[:8]}... ({next((m['model_class_key'] for m in models if m['id'] == x), '')})"
    )
    
    # Информация о модели
    if selected_model_id:
        model_info = get_model_info(selected_model_id)
        if model_info:
            st.warning(f"**Внимание!** Вы собираетесь удалить модель:")
            st.write(f"- **ID:** `{model_info['id']}`")
            st.write(f"- **Класс:** {model_info['model_class_key']}")
            st.write(f"- **Статус:** {model_info['status']}")
            st.write(f"- **Создана:** {model_info['created_at']}")
    
    # Тип удаления
    hard_delete = st.checkbox(
        "Полное удаление (удалить файлы с диска)",
        help="Если не отмечено, модель будет только помечена как удалённая"
    )
    
    # Кнопка удаления
    if st.button("🗑️ Удалить модель", type="primary"):
        if delete_model(selected_model_id, hard=hard_delete):
            delete_type = "полностью удалена" if hard_delete else "помечена как удалённая"
            st.success(f"✅ Модель {delete_type}!")
            st.balloons()
            
            # Небольшая задержка и перезагрузка
            import time
            time.sleep(1)
            st.rerun()


# ============================================================================
# Футер
# ============================================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>ML Service Dashboard v1.0.0 | Powered by Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
