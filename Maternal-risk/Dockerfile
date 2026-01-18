FROM python:3.12.1-slim-bookworm

RUN pip install uv

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /code

ENV PATH="/code/.venv/bin:$PATH"

COPY "pyproject.toml" "uv.lock" ./

RUN uv sync --frozen

COPY "predict.py" "MR_model.bin" ./

EXPOSE 10000

CMD ["sh", "-c", "uvicorn predict:app --host 0.0.0.0 --port ${PORT}"]