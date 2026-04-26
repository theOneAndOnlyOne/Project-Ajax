FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Persistent supplement log lives here. Mount a volume for /app/data
# if you want it to survive container restarts.
RUN mkdir -p /app/data

EXPOSE 8080
ENV PORT=8080 LEON_HOST=0.0.0.0

# 2 workers is plenty for a single-user dashboard.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--timeout", "30", "app:app"]
