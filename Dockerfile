# Multi-stage build to avoid cache issues
FROM python:3.9-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    zstd \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Python dependencies
RUN pip install --no-cache-dir \
    torch==2.0.1 \
    torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
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

# Production stage
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set working directory
WORKDIR /app

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
CMD ["sh", "-c", "ollama serve & python app_heroku.py"]
