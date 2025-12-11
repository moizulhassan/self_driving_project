 Self-Driving Vehicle Detection System (YOLOv8 + OpenCV)

This project is a real-time **Self-Driving Vehicle Detection System** built using **YOLOv8**, **OpenCV**, and **Python**.  
It detects vehicles such as cars, trucks, buses, and motorcycles from video files or a live webcam stream.  
The goal of this project is to demonstrate the core perception module used in autonomous driving systems.

---

Project Summary

Modern self-driving cars rely heavily on computer vision to understand their surroundings.  
This project replicates that concept by using a YOLO deep-learning model to:

- Detect vehicles frame-by-frame  
- Draw bounding boxes in real time  
- Process live or saved video files  
- Produce high-accuracy detections even on low-performance hardware

This project showcases:
- Deep learning (YOLOv8)
- Image processing (OpenCV)
- Real-time inference
- Autonomous vehicle perception basics

---

## 📂 Project Structure

self_driving_project/
│
├── src/
│ ├── detector.py # YOLO model wrapper
│ └── run_demo.py # Main execution script
│
├── yolov8n.pt # YOLO weights file
│
├── data/
│ └── videos/ # Test videos (MP4)
│
├── requirements.txt
└── README.md   


---

## ⚙️ Installation & Setup

Follow these steps to run the project on your machine:

### 1️⃣ Create a virtual environment
```bash
python -m venv venv
.\venv\Scripts\activate

Install all dependencies
pip install -r requirements.txt


---

## ⚙️ Installation & Setup

Follow these steps to run the project on your machine:

### 1️⃣ Create a virtual environment
```bash
python -m venv venv
.\venv\Scripts\activate

Add YOLOv8 weights

Place yolov8n.pt in the project root:
  
C:\self_driving_project\yolov8n.pt   

Add test videos 
self_driving_project/data/videos/

How to Run the System 
python src/run_demo.py ".\data\videos\your_video.mp4"
 
Run using webcam 
python src/run_demo.py 0
 
How It Works (Simple Explanation)

The YOLO model loads into memory

Video frames are captured one-by-one

YOLO detects vehicles in each frame

Bounding boxes + labels are drawn

The processed frames are shown in a live window

This structure represents the core vision pipeline in autonomous vehicles.