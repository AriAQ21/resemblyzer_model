FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
COPY diarization.py .

# Install system build dependencies needed for webrtcvad and others
RUN apt-get update && apt-get install -y \
    gcc \
    make \
    libffi-dev \
    libsndfile1-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python", "diarization.py", "--help"]
