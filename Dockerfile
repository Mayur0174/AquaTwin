# AquaTwin AI - single image for API, dashboard and batch jobs (CPU)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU-only PyTorch wheel keeps the image small
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -r requirements.txt

COPY aquatwin ./aquatwin
COPY dashboard ./dashboard
COPY configs ./configs
COPY tests ./tests
COPY pyproject.toml README.md ./
RUN pip install --no-deps -e .

EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s CMD curl -fs http://localhost:8000/health || curl -fs http://localhost:8501/_stcore/health || exit 1
CMD ["uvicorn", "aquatwin.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
