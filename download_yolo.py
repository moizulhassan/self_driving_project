# download_yolo.py
from ultralytics import YOLO
print("Downloading yolov8n weights (this may take a minute)...")
YOLO("yolov8n.pt")   # this will download the weight if not present
print("Done.")
