FROM python:3.12-slim

# Prevent python buffering
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file first (better caching)
COPY requirements-docker.txt .

RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy project
COPY . .

# Expose API port
EXPOSE 8001

# Start FastAPI backend
CMD ["uvicorn", "src.api.app_best:app", "--host", "0.0.0.0", "--port", "8001"]