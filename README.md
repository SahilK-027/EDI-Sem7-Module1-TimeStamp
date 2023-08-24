# Person Detection from Video using YOLOv8

This script performs person detection on a video using the YOLOv8 object detection model. It detects instances of the "person" class with a confidence threshold of 0.8 and records their timestamps in a JSON file.

## Prerequisites

- Python 3.x
- OpenCV (`cv2`)
- `ultralytics` library (YOLOv8)
- `math`, `time`, `pathlib`, `json` modules

## Usage

1. Clone the repository:

   ```bash
    git clone <repository_url>
    cd <repository_name>
   ```
2. Set up a virtual environment:

    ```bash
    python3 -m venv venv
    ```
    - On mac / linux
    ```bash
    source venv/bin/activate
    ```
    - On windows
    ```bash
    venv\Scripts\activate
    ```


3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the script:
    ```bash
    python main.py
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