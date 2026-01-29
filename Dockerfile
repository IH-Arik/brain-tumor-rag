FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first
COPY requirements.slim.txt requirements.txt

# Install only essential Python packages
RUN pip install --no-cache-dir \
    flask==2.3.3 \
    torch==2.0.1 \
    torchvision==0.15.2 \
    pillow==10.0.0 \
    sentence-transformers==2.2.2 \
    scikit-learn==1.3.0 \
    faiss-cpu==1.7.4 \
    transformers==4.30.2 \
    numpy==1.24.3

# Copy only essential application files
COPY app_heroku.py .
COPY knowledge_base.py .
COPY vector_store.py .
COPY rag_engine.py .

# Create necessary directories
RUN mkdir -p uploads static templates

# Copy templates
COPY templates/ templates/

# Create labels.txt if it doesn't exist
RUN echo -e "glioma_tumor\nmeningioma_tumor\nno_tumor\npituitary_tumor" > labels.txt

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
