# CPU image for the dashboard + pipeline.
#   docker build -t redlight .
#   docker run -p 8000:8000 -v $PWD/data:/data redlight
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt
COPY backend ./backend
COPY tools ./tools
COPY evaluation ./evaluation
ENV RLV_DATA_DIR=/data
WORKDIR /app/backend
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
