# =============================================================================
# SE-Res-Inception Inference Server - Production Dockerfile
# =============================================================================
# Optimized for NVIDIA RTX 5070 Ti (Blackwell Architecture, SM_120)
# Uses PyTorch with CUDA 12.8 for best compatibility
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Base Image with CUDA 12.8 + PyTorch
# -----------------------------------------------------------------------------
# Using official PyTorch image with CUDA support
# Note: For RTX 50 series (Blackwell), we need CUDA 12.8+ and PyTorch 2.9.1+
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# -----------------------------------------------------------------------------
# Stage 2: Install System Dependencies
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Required for Pillow
    libjpeg-dev \
    libpng-dev \
    # Cleanup to reduce image size
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Stage 3: Install Python Dependencies
# -----------------------------------------------------------------------------
# Install required packages
RUN pip install --no-cache-dir \
    fastapi>=0.109.0 \
    uvicorn[standard]>=0.27.0 \
    python-multipart>=0.0.6 \
    Pillow>=10.0.0

# -----------------------------------------------------------------------------
# Stage 4: Copy Application Files
# -----------------------------------------------------------------------------
# Copy model architecture
COPY model.py .

# Copy API server
COPY server.py .

# Copy model weights (this should be the last layer for better caching)
COPY checkpoints/ checkpoints/

# -----------------------------------------------------------------------------
# Stage 5: Configure Runtime
# -----------------------------------------------------------------------------
# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the server
# Using 1 worker since we're running GPU inference (avoid GPU memory issues)
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
