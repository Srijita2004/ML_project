FROM python:3.11-slim

# Install system dependencies for OpenCV and image decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and install CPU-optimized PyTorch
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy dependencies and install
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Copy application files and model checkpoints
COPY app_final.py .
COPY road_expanded_best.pt .
COPY road_best.pt .
COPY fall_expanded_best.pt .
COPY fall_accident_model_best.pt .

# Set default port to 7860 (works for Hugging Face Spaces; Render/Railway pass their own PORT env)
ENV PORT=7860 \
    HOST=0.0.0.0 \
    PYTHONUNBUFFERED=1

EXPOSE 7860

# Start with Gunicorn WSGI production server
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-7860} --workers 1 --threads 4 --timeout 120 app_final:app"]
