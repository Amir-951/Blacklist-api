FROM python:3.10-slim

# Install system dependencies for dlib/face_recognition
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy app files
COPY app.py .
COPY data/ ./data/

EXPOSE 8000

CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
