FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for Docker layer caching
COPY requirements_space.txt .
RUN pip install --no-cache-dir -r requirements_space.txt

# Copy the full project
COPY . .

# Create results dir
RUN mkdir -p results

# HuggingFace Spaces runs on port 7860
EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
