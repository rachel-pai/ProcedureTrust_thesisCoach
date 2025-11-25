# Use a lightweight Python image
FROM python:3.12-slim

# Avoid Python buffering issues
ENV PYTHONUNBUFFERED=1

# Set workdir
WORKDIR /app

# Install system deps (optional but often needed for psycopg2, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt /app/requirements.txt

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY app.py /app

# Streamlit runs on 8501 by default
EXPOSE 8501

# Lightsail will inject a PORT env var; fall back to 8501 if not set
ENV PORT=8501

# Entry command
CMD streamlit run app.py \
    --server.port=$PORT \
    --server.address=0.0.0.0
