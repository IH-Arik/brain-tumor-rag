# Use Ubuntu base for Ollama compatibility
FROM ubuntu:22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Python dependencies
RUN python3 -m pip install --no-cache-dir \
    torch==2.0.1+cpu \
    torchvision==0.15.2+cpu \
    flask==2.3.3 \
    flask-cors==4.0.0 \
    pillow==10.0.0 \
    timm==0.9.7 \
    numpy==1.24.3 \
    requests==2.31.0 \
    langchain==0.0.340 \
    faiss-cpu==1.7.4 \
    sentence-transformers==2.2.2 \
    huggingface-hub==0.19.4

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt ./

# Copy application files
COPY app_ollama.py app_heroku.py

# Create necessary directories
RUN mkdir -p uploads static

# Copy static files
COPY static/ static/
COPY templates/ templates/

# Create labels file
RUN printf "glioma_tumor\nmeningioma_tumor\nno_tumor\npituitary_tumor\n" > labels.txt

# Expose port
EXPOSE 5000

# Start Ollama in background and run the app
CMD ["sh", "-c", "ollama serve & python3 app_heroku.py"]
