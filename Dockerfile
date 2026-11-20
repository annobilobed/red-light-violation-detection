# CPU image: dashboard + pipeline + evaluation scripts.
#
#   docker build -t redlight .
#   docker run -p 8000:8000 -v "$PWD/data:/data" redlight
#   open http://localhost:8000
#
# No system packages are needed: ffmpeg comes from imageio-ffmpeg and OpenCV
# is the headless build. Build arguments:
#   BASE_IMAGE   e.g. mirror.gcr.io/library/python:3.12-slim if Docker Hub
#                rate-limits you
#   TORCH_INDEX  PyTorch wheel index; the default CPU index keeps the image
#                small. Set it to "" to use PyPI (larger, includes CUDA).
# Behind a TLS-inspecting proxy, pass its CA as a build secret (it is only
# mounted while downloading and never stored in the image):
#   docker build --secret id=ca,src=/path/to/ca.crt -t redlight .
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt .
RUN --mount=type=secret,id=ca,required=false \
    if [ -f /run/secrets/ca ]; then export PIP_CERT=/run/secrets/ca SSL_CERT_FILE=/run/secrets/ca \
      REQUESTS_CA_BUNDLE=/run/secrets/ca; fi \
 && if [ -n "$TORCH_INDEX" ]; then pip install torch torchvision --index-url "$TORCH_INDEX"; fi \
 && pip install -r requirements.txt \
 # Swap the desktop OpenCV pulled in by Ultralytics for the headless build
 # (same cv2 module, no libGL needed).
 && pip uninstall -y opencv-python \
 && pip install "opencv-python-headless>=4.9,<5"

COPY backend ./backend
COPY tools ./tools
COPY evaluation ./evaluation

# Bake models in so the container works offline: YOLO11n (vehicles, people)
# and the checksum-pinned OpenCV face detector used for anonymisation.
RUN --mount=type=secret,id=ca,required=false \
    if [ -f /run/secrets/ca ]; then export SSL_CERT_FILE=/run/secrets/ca REQUESTS_CA_BUNDLE=/run/secrets/ca; fi \
 && python tools/download_models.py \
 && cd backend && python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

ENV RLV_DATA_DIR=/data YOLO_CONFIG_DIR=/tmp/Ultralytics
VOLUME /data
WORKDIR /app/backend
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health', timeout=4)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
