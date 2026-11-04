# syntax=docker/dockerfile:1.7
# AquaTwin AI - single image for API, dashboard and batch jobs (CPU)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# All dependencies ship as manylinux wheels (xgboost and scikit-learn bundle OpenMP), so no apt layer is needed.
COPY requirements.txt .
# CPU-only PyTorch keeps the image small; falls back to PyPI if the PyTorch index is unreachable.
# Behind a TLS-inspecting proxy, pass its CA without baking it into the image:
#   docker build --secret id=extra_ca,src=/path/to/ca.crt .
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN --mount=type=secret,id=extra_ca \
    if [ -s /run/secrets/extra_ca ]; then export PIP_CERT=/run/secrets/extra_ca; fi; \
    (pip install --index-url "$TORCH_INDEX" torch || pip install torch) \
    && pip install -r requirements.txt

COPY aquatwin ./aquatwin
COPY dashboard ./dashboard
COPY configs ./configs
COPY tests ./tests
COPY ui ./ui
COPY pyproject.toml README.md ./
# the package runs from source: no install step, no network needed after the dependency layer
ENV PYTHONPATH=/app

EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD ["python", "-m", "aquatwin.healthcheck"]
CMD ["uvicorn", "aquatwin.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
