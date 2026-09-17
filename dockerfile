FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements/ ./requirements/
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt || pip install --no-cache-dir -r requirements/requirements.txt

# Copy all project code, frontend, scripts, and model binaries
COPY . .

# Expose Hugging Face Space default port
EXPOSE 7860

# Start FastAPI server
CMD ["uvicorn", "API.main:app", "--host", "0.0.0.0", "--port", "7860"]