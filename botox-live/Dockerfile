# Base image with CUDA 12.1 and PyTorch pre-installed
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Set working directory inside container
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8003

# Use multiple workers for better performance
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003", "--workers", "4"] 