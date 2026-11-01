# syntax=docker/dockerfile:1
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DATA_DIR=/data \
    RESULTS_DIR=/results DOWNLOAD_DATA=false TORCH_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    API_PORT=9000
WORKDIR /opt/dfl
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
COPY app ./app
COPY experiments ./experiments
COPY scripts ./scripts
ARG DATASET
RUN --mount=type=cache,target=/dataset-cache,sharing=locked \
    python -m scripts.experiment_tools.prepare_data --trainer-module "experiments.$DATASET.trainer" --data-dir "/dataset-cache/$DATASET" \
    && mkdir -p /data /results \
    && if [ "$DATASET" = "harth" ]; then \
         mkdir -p /data/harth && cp -a /dataset-cache/harth/harth/prepared /data/harth/prepared; \
       else cp -a "/dataset-cache/$DATASET/." /data/; fi
EXPOSE 9000
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port \"$API_PORT\""]
