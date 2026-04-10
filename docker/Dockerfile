FROM python:3.10-slim

# Install system dependencies for OpenCV and general runtimes
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
