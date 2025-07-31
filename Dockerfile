FROM python:3.11-slim

WORKDIR /app

# Upgrade pip to latest version
RUN pip install --upgrade pip

# Copy and install requirements
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

COPY . .

CMD ["python", "main.py"] 