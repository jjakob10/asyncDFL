FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/data \
    RESULTS_DIR=/results \
    DOWNLOAD_DATA=false \
    TORCH_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    TRAINER_MODULE=experiments.ton_iot.trainer

WORKDIR /opt/dfl

RUN apt-get update \
    && apt-get install --no-install-recommends -y iproute2 iputils-ping \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY experiments/__init__.py ./experiments/__init__.py
COPY experiments/ton_iot ./experiments/ton_iot

RUN mkdir -p /data/ton_iot /results

EXPOSE 8000
VOLUME ["/results"]

HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=6 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
