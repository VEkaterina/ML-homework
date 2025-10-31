FROM python:3.13.5-slim-bookworm

FROM agrigorev/zoomcamp-model:2025

COPY "pipeline_v2.bin" ./

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN python -m pip install --no-cache-dir fastapi uvicorn requests scikit-learn

COPY "pyproject.toml" "uv.lock" ./

RUN uv sync --locked

COPY "predict.py" ./

EXPOSE 9696

ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]