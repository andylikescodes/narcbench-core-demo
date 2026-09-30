# NARCBench Core demo — Zeabur / PaaS presenter (Python API + static viewer; no GPU)
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PORT=8080 \
    PYTHONDONTWRITEBYTECODE=1

# Stdlib-only server; requirements kept for Zeabur Python detection / future deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt || true

COPY app/ ./app/
COPY scripts/ ./scripts/
COPY data/ ./data/
COPY upstream/ ./upstream/

# Refresh indexes so source_dir paths match WORKDIR (/app), not the build host
RUN python3 scripts/build_index.py

EXPOSE 8080

# Honor PORT (Zeabur injects it); server binds 0.0.0.0
CMD ["python3", "app/server.py"]
