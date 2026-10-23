# Ethiopian Coffee Doctor — single-service image (FastAPI API + static frontend).
# Builds small and runs on any free container host (Render, Railway, Fly.io,
# Hugging Face Spaces "Docker" SDK, etc.) with no GPU required.

FROM python:3.11-slim

# System deps required by Pillow / TensorFlow-CPU wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

# Install Python dependencies first (better layer caching)
COPY backend/requirements.txt backend/setup.py ./backend/
COPY backend/src ./backend/src
RUN pip install --no-cache-dir -r backend/requirements.txt

# App code
COPY backend/app ./backend/app
COPY backend/main.py ./backend/main.py
COPY backend/final_model ./backend/final_model
COPY frontend ./frontend

ENV PORT=8000
ENV SERVE_FRONTEND=1
ENV PYTHONUNBUFFERED=1

WORKDIR /srv/backend
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
