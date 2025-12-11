FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Create cache directory for HuggingFace models
RUN mkdir -p /root/.cache/huggingface

EXPOSE 50051

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:50051/health || exit 1

CMD ["python", "mcp_server.py"]


