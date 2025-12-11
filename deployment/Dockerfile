FROM langchain/langchain

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install uv 
RUN uv pip install -r requirements.txt --system


# RUN uv pip install mcp acp-sdk

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--workers", "1"]

EXPOSE 8000

# tag: v0.0.6
# image-name: gst_services