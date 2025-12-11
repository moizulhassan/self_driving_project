import cv2
import sys
from detector import YOLODetector

def main(path):
    # Initialize detector with local weights
    detector = YOLODetector(weights_path="C:/self_driving_project/yolov8n.pt", conf=0.35)

    # Check if input is webcam or video
    if path == "0":
        cap = cv2.VideoCapture(0)  # webcam
    else:
        cap = cv2.VideoCapture(path)  # video file

    if not cap.isOpened():
        print("Error opening video source")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection
        results = detector.detect(frame)

        # Draw bounding boxes
        frame = detector.draw_boxes(frame, results)

        # Show
        cv2.imshow("YOLOv8 Demo", frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Usage: python run_demo.py [0 for webcam or path to video]")
