FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    fastapi==0.104.0 \
    "uvicorn[standard]==0.24.0" \
    grpcio==1.59.0 \
    grpcio-tools==1.59.0 \
    streamlit==1.28.0 \
    pandas==2.0.0 \
    scikit-learn==1.3.0 \
    catboost==1.2 \
    joblib==1.3.0 \
    pydantic==2.4.0 \
    requests==2.31.0

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
