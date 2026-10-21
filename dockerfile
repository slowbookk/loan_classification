# Use the same Python version you used for training.
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py .
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

CMD ["sh", "-c", "exec uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]