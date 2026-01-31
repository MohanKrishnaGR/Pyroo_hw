# Base image for Edge AI (This would ideally be an NVIDIA L4T base on Jetson)
# For this standard implementation, we use Python 3.11-slim
FROM python:3.11-slim AS base

# Install system dependencies for OpenCV and GStreamer
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer1.0-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Stage 2: Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Environment setup
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run the pipeline (Default command)
CMD ["python", "deployment/fire_detection_pipeline.py"]
