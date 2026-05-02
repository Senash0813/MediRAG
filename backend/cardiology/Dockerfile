FROM python:3.11-slim-bookworm

WORKDIR /app

# Install Python dependencies first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy local model files
# instructor-large, inbedder-roberta-large, projector.pt, HyDE model
# Phi model is NOT copied into the image.
# It will be downloaded from HuggingFace during FastAPI startup.
COPY models/ ./models/

# Copy FAISS vectorstore index
COPY vectorstore/ ./vectorstore/

# Copy application code
COPY *.py ./

# Expose FastAPI port
EXPOSE 8001

# HuggingFace cache directory
# Use /workspace so RunPod can persist downloaded HuggingFace models
# without baking Phi into the Docker image.
ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface

# Ollama settings
# Connect to Ollama service (via docker-compose or external Ollama instance)
# Default points to ollama service in docker-compose; override with host network if needed
ENV OLLAMA_BASE_URL=http://ollama:11434

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]