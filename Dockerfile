FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create requirements.txt directly
RUN printf "flask==2.3.3\ntorch==2.0.1\ntorchvision==0.15.2\npillow==10.0.0\nsentence-transformers==2.2.2\nscikit-learn==1.3.0\nfaiss-cpu==1.7.4\ntransformers==4.30.2\nnumpy==1.24.3\n" > requirements.txt

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

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
