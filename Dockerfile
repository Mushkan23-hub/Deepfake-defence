FROM python:3.11-slim

# Install system dependencies
# exiftool, ffmpeg (includes ffprobe), libsndfile for audio
RUN apt-get update && apt-get install -y \
    libexiftool-perl \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Make exiftool available as 'exiftool'
RUN ln -s /usr/bin/exiftool /usr/local/bin/exiftool || true

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Create uploads folder
RUN mkdir -p uploads

# Expose port
EXPOSE 5000

# Run with gunicorn (production server)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "300", "app:app"]
