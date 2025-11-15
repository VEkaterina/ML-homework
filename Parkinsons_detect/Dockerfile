FROM python:3.12.1-slim-bookworm

RUN pip install uv

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /code

ENV PATH="/code/.venv/bin:$PATH"

COPY "pyproject.toml" "uv.lock" ".python-version" ./

RUN uv sync --frozen

COPY "predict.py" "PD_model.bin" ./

ENV PORT=10000

EXPOSE 10000

CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "10000"]