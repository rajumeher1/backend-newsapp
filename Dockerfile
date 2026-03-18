FROM python:3.12-slim

WORKDIR /runner

# Install system deps (optional, if needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of code
COPY . .

CMD ["python", "-m", "cron.main"]