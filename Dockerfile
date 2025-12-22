# Use a slim python image to save space
FROM python:3.10-slim

# Install system dependencies required for Pillow/OpenCV if needed
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your python code (assuming your file is named main.py)
COPY main.py .

# Expose the port
EXPOSE 8000

# Run the app
CMD ["python", "main.py"]
