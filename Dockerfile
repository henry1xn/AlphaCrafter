# AlphaCrafter — reproducible runs (LLM + Yahoo + SQLite under /app/data)
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY pyproject.toml README.md requirements.txt ./
COPY alphacrafter ./alphacrafter
COPY scripts ./scripts

RUN pip install -U pip setuptools wheel \
    && pip install -r requirements.txt \
    && pip install -e .

# Default: show help: override command in docker run / compose
CMD ["python", "scripts/run_alphacrafter.py", "--help"]
