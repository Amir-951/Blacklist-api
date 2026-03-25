FROM python:3.10-slim

# Runtime libs only — no cmake, no compilation
RUN apt-get update && apt-get install -y \
    libopenblas0 \
    liblapack3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
# dlib-bin = pre-compiled dlib (no cmake needed)
# face_recognition installed --no-deps so it won't try to recompile dlib
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install dlib-bin && \
    pip install face_recognition_models && \
    pip install face_recognition --no-deps && \
    pip install fastapi "uvicorn[standard]" Pillow numpy python-multipart

# Create required directories (static/ for old UI, data/blacklist/ for images)
RUN mkdir -p /app/static /app/data/blacklist /app/data/users

# Copy app files
COPY app.py .
COPY data/metadata.json ./data/

EXPOSE 8000

CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
