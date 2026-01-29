FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create requirements.txt with CPU-only versions
RUN printf "flask==2.3.3\ntorch==2.0.1+cpu\ntorchvision==0.15.2+cpu\npillow==10.0.0\nsentence-transformers==2.2.2\nscikit-learn==1.3.0\nfaiss-cpu==1.7.4\ntransformers==4.30.2\nnumpy==1.24.3\n" > requirements.txt

# Install Python packages with CPU-only PyTorch and minimal dependencies
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.0.1+cpu \
    torchvision==0.15.2+cpu \
    && pip install --no-cache-dir \
    flask==2.3.3 \
    flask-cors==4.0.0 \
    pillow==10.0.0 \
    timm==0.9.7 \
    numpy==1.24.3 \
    requests==2.31.0

# Copy only essential application files
COPY app_llm.py app_heroku.py

# Create necessary directories and copy static files
RUN mkdir -p uploads static

# Copy static files explicitly
COPY static/ static/

# Copy templates
COPY templates/ templates/

# Create labels.txt
RUN printf "glioma_tumor\nmeningioma_tumor\nno_tumor\npituitary_tumor\n" > labels.txt

# Create uploads directory
RUN mkdir -p uploads

# Expose port
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run the application
CMD ["python", "app_heroku.py"]
