FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl && \
    rm -rf /var/lib/apt/lists/*

# Ставим Poetry
# RUN curl -sSL https://install.python-poetry.org | python3 -
# бывает что курл может лежать мертвым
RUN python -m pip install --no-cache-dir "poetry>=2.0.0" && poetry --version


# Создаем окружение при помощи poetry
RUN poetry config virtualenvs.create false \
 && poetry config virtualenvs.in-project false \
 && poetry config virtualenvs.path /opt/venv

 # Cтавим зависимости через poetry
COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-interaction --no-ansi --no-root

COPY app/ ./app/
COPY grpc_service/ ./grpc_service/
COPY rest_api.py ./
COPY dashboard.py ./

RUN python -m grpc_tools.protoc \
    -I. \
    --python_out=. \
    --grpc_python_out=. \
    grpc_service/ml_service.proto

RUN mkdir -p /app/storage/models /app/storage/metadata

ENV PYTHONPATH="/app/app:${PYTHONPATH}"

EXPOSE 8000 50051 8501
