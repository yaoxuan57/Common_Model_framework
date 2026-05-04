# PyTorch + CUDA 12.1 + cuDNN runtime (includes torch already)
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Good defaults
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

# (Optional) system tools

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates gnupg && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python deps (DO NOT include torch here; it's already in the base image) ---
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

# Your code
COPY . /app

# Default command (you can override in docker run)
CMD ["python", "Unifault_release/fine_tune.py"]
