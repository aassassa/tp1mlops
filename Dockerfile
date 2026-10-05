FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (build-essential, curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and default data directory
COPY src/ ./src/
COPY data/ ./data/

# Expose the port FastAPI runs on
EXPOSE 8000

# Set Python Path and host configs
ENV PYTHONPATH=/app
ENV HOST=0.0.0.0
ENV PORT=8000

# Start FastAPI API using uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
