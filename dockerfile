# 1. Basic image with Python
FROM python:3.10-slim

# 2. Environment setting
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Working directory inside container
WORKDIR /app

# 4. First copy only requirements
COPY requirements.txt .

# 5. Download requirements
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy initial app and model
COPY app ./app
COPY model ./model

# 7. Open port for server
EXPOSE 8000

# 8. Command to run FastAPI via uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]