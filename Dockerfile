# FinBERT sentiment API (CPU). Build: docker build -t finbert .  Run: docker run -p 8000:8000 finbert
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_NO_CACHE=1 \
    HF_HOME=/opt/hf-cache \
    FINBERT_MODEL=ProsusAI/finbert

WORKDIR /app

# CPU-only torch keeps the image small (the PyPI default wheel bundles CUDA).
RUN uv pip install "torch>=2.4" --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN uv pip install .

# Bake the model into the image so containers start without network access.
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='${FINBERT_MODEL}')"

RUN useradd --create-home app && chown -R app /opt/hf-cache
USER app

EXPOSE 8000
CMD ["uvicorn", "finbert.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
