# Use official Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ../models /app/models
COPY inference/infer.py /app/infer.py

COPY ../data /app/data

# Create output directory
RUN mkdir -p /app/output

# Run inference script
CMD ["python", "infer.py"]