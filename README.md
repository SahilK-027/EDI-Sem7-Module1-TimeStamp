# Person Detection from Video using YOLOv8

This script performs person detection on a video using the YOLOv8 object detection model. It detects instances of the "person" class with a confidence threshold of 0.8 and records their timestamps in a JSON file.

## Prerequisites

- Python 3.x
- OpenCV (`cv2`)
- `ultralytics` library (YOLOv8)
- `math`, `time`, `pathlib`, `json` modules

## Usage

1. Install required packages using `pip`:

   ```bash
   pip install opencv-python-headless ultralytics
   ```
2. Save the YOLOv8 model checkpoint (e.g., yolov8n.pt) in the same directory.
3. Place the input video (video.mp4) in the same directory.
4. Run the script:
    ```bash
    python script.py
    ```
5. Press q to exit the detection process.


## Results
Detected "person" instances and their timestamps are stored in detection_results.json in the following format:
```json
[
    [
        {
            "class": "person",
            "confidence": 0.91,
            "timestamp": 3.4034
        }
    ],
    [
        {
            "class": "person",
            "confidence": 0.9,
            "timestamp": 4.4044
        }
    ]
]
```