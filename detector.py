from ultralytics import YOLO
import cv2

class YOLODetector:
    def __init__(self, 
                 weights_path="C:/self_driving_project/yolov8n.pt",
                 device="cpu",
                 conf=0.4):

        # Load YOLO model from local file
        self.model = YOLO(weights_path)

        # Confidence threshold
        self.conf = conf

        # Device (cpu or cuda)
        self.device = device

    def detect(self, image):
        """
        Run detection on an OpenCV image (numpy array)
        Returns YOLO predictions
        """
        results = self.model(image, conf=self.conf, device=self.device)
        return results

    def draw_boxes(self, image, results):
        """
        Draw bounding boxes on image
        """
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = f"{self.model.names[cls]} {conf:.2f}"

                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 255, 0), 2)
                cv2.putText(image, label, (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return image
