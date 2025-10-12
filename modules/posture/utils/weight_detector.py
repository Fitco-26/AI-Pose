import cv2
from ultralytics import YOLO
import torch

# Explicitly set the device to leverage GPU if available.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Load YOLOv8 small pretrained model
# small, fast; can switch to yolov8s.pt for more accuracy
model = YOLO("yolov8n.pt").to(device)


def detect_weight(frame, wrist_coords):
    """
    Detect if a weight (dumbbell) is present near the wrist.
    frame: full image
    wrist_coords: (x, y) normalized (0-1) from Mediapipe
    """
    h, w, _ = frame.shape
    x, y = int(wrist_coords[0] * w), int(wrist_coords[1] * h)

    # Crop region around wrist
    x1, y1 = max(0, x - 80), max(0, y - 80)
    x2, y2 = min(w, x + 80), min(h, y + 80)
    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return False

    # Run YOLO detection on cropped region
    results = model.predict(roi, imgsz=128, verbose=False)

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            # You may replace "sports ball" with your custom dumbbell class after training
            if label in ["sports ball", "person"]:
                return True

    return False
