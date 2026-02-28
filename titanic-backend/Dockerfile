# Use the official Python 3.12 slim image based on .python-version
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies if required by standard data science or ML packages
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install PyTorch EXPLICITLY first (CPU version to save size and time)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port
EXPOSE 8000

# Command to run the application
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]
