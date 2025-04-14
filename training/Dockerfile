# Use official Python image
FROM python:3.9

# Set working directory
WORKDIR /app

## Copy requirements and install
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy project files
COPY training/train.py train.py
COPY ../data /data

# Create models directory
RUN mkdir -p /app/models

# Run training script
CMD ["python", "train.py"]


