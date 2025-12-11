# utils.py
import cv2

def draw_box(frame, bbox, track_id=None, label=None, color=(0,255,0)):
    x1,y1,x2,y2 = bbox
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
    text = f"ID:{track_id}" if track_id is not None else label
    if label:
        text = f"{label}"
    cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
