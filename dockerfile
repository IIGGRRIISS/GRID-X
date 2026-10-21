FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch first, so pip doesn't pull the CUDA build
RUN pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Download the models from Hugging Face during the build
RUN python download_models.py

EXPOSE 7860
CMD ["uvicorn", "API.main:app", "--host", "0.0.0.0", "--port", "7860"]