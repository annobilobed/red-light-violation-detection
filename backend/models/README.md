Optional model files for anonymisation. Fetch them with

    python tools/download_models.py

(OpenCV res10 SSD face detector, checksum-pinned). Without them the face layer
falls back to OpenCV's Haar cascades. Put fine-tuned YOLO checkpoints for
analysis in backend/data/models/ instead.
