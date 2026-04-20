FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

# Keeps Python output unbuffered and prevents .pyc files
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Point HuggingFace cache to the RunPod network volume
# The volume must be mounted at /runpod-volume in the RunPod console
ENV HF_HOME=/runpod-volume/hf_cache

WORKDIR /app

# Minimal system deps (git needed by some HF packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# torch is already in the base image — requirements.txt has it commented out
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK data so BM25Retriever never calls the network at runtime
RUN python -c "import nltk; nltk.download('punkt_tab', quiet=True)"

# Copy application code, local model weights, and pre-built indices
COPY . .

EXPOSE 8000

# Single worker — multiple workers would load all models multiple times
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
