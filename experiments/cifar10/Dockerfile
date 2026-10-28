FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/data \
    RESULTS_DIR=/results \
    DOWNLOAD_DATA=false \
    TORCH_NUM_THREADS=1 \
    OMP_NUM_THREADS=1

WORKDIR /opt/dfl

RUN apt-get update \
    && apt-get install --no-install-recommends -y iproute2 iputils-ping \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY experiments/__init__.py ./experiments/__init__.py
COPY experiments/cifar10 ./experiments/cifar10
COPY experiments/johann ./experiments/johann

ENV API_PORT=9000

ENV TRAINER_MODULE=experiments.cifar10.trainer

RUN mkdir -p /data /results \
    && python -c "from torchvision.datasets import CIFAR10; CIFAR10.url = 'https://huggingface.co/datasets/xingslong/cifar-10-batches-py/resolve/main/cifar-10-python.tar.gz'; CIFAR10('/data', train=True, download=True); CIFAR10('/data', train=False, download=True)"

EXPOSE 9000
VOLUME ["/results"]

HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=6 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ['API_PORT'] + '/health', timeout=2)"

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port \"$API_PORT\""]
