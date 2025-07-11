FROM python:3.9-slim

WORKDIR /app

# Copy code and requirements
COPY requirements.txt .
COPY inference.py .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Default command shows usage, override at runtime
CMD ["python", "inference.py", "--help"]
