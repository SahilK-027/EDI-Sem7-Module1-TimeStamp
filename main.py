import cv2
import time
from ultralytics import YOLO
import math
from pathlib import Path
import json

# Load YOLOv5 model
model = YOLO('yolov8n.pt')

# Load video
video_path = 'video.mp4'
cap = cv2.VideoCapture(video_path)

detection_results = []  # List to store detection results

# Define class names
class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
             'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
             'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
             'ring', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
             'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
             'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
             'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
             'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
             'teddy bear', 'hair drier', 'toothbrush' ]

initial_timestamp = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    start_time = time.time()
    
    # Perform person detection
    results = model(frame)

    # Perform person detection
    detections = []
    
    for i in results:
        boundingBoxes = i.boxes
        for box in boundingBoxes:
            confidence = math.ceil(box.conf[0] * 100) / 100
            classIdx = int(box.cls[0])
            currentClass = class_names[classIdx]
            if currentClass == 'person' and confidence > 0.8:
                current_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if(initial_timestamp == 0):
                    detections.append({
                        "class": currentClass,
                        "confidence": confidence,
                        "timestamp": current_timestamp
                    })
                    detection_results.append(detections)
                    initial_timestamp = current_timestamp
                else :    
                    delta_time = current_timestamp - initial_timestamp
                    if(delta_time > 1):
                        detections.append({
                            "class": currentClass,
                            "confidence": confidence,
                            "timestamp": current_timestamp
                        })
                        detection_results.append(detections)
                        initial_timestamp = current_timestamp
                
    end_time = time.time()
    fps = 1.0 / (end_time - start_time)
    
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("Person Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# Save results to a JSON file
with open('detection_results.json', 'w') as json_file:
    json.dump(detection_results, json_file)