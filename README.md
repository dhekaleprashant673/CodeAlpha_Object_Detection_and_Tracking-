# Real-time Object Detection and Tracking

This project implements a real-time object detection and tracking system using OpenCV and YOLO. It can process video from a webcam or a video file, detect objects using a pre-trained YOLO model, and track them using a simplified implementation of the SORT (Simple Online and Realtime Tracking) algorithm.

## Features

- Real-time video input from webcam or video file
- Object detection using YOLOv8 pre-trained model
- Object tracking with unique IDs using SORT algorithm
- Display of bounding boxes, labels, tracking IDs, and FPS

## Requirements

- Python 3.8 or higher
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository or download the files

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Download a pre-trained YOLOv8 model (the script will automatically download YOLOv8n if not present)

## Usage

### Quick Start

The easiest way to run the application is with the main script, which will create a sample video and process it:

```bash
python object_detection_tracking.py
```

This will generate an output video file `output_detection.mp4` with object detection and tracking results.

#### Extending the Maximum Frame Count

By default, the application processes up to 300 frames. You can increase this limit using the `--max-frames` parameter:

```bash
python object_detection_tracking.py --max-frames 600
```

This will process up to 600 frames instead of the default 300.

### Using Webcam

To run the application using your webcam as the video source:

```bash
python run_with_webcam.py
```

You can also specify the maximum number of frames to process:

```bash
python run_with_webcam.py --max-frames 600
```

### Using a Video File

To run the application using a video file as the source, you can use the provided `run_with_video.py` script:

```bash
python run_with_video.py path/to/your/video.mp4
```

You can also specify the maximum number of frames to process:

```bash
python run_with_video.py path/to/your/video.mp4 --max-frames 600
```

### Creating Sample Video

If you want to create a sample video for testing:

```bash
python sample_video.py
```

### Controls

- Press 'q' to quit the application

## Customization

### Changing the YOLO Model

You can use different YOLOv8 models by changing the model_path parameter:

```python
# For a larger, more accurate model
detector = ObjectDetectionTracking(source=0, model_path="yolov8m.pt")

# For a smaller, faster model
detector = ObjectDetectionTracking(source=0, model_path="yolov8n.pt")
```

### Adjusting Tracking Parameters

You can adjust the tracking parameters by modifying the Sort class initialization:

```python
# Modify these values for different tracking behavior
self.tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
```

- `max_age`: Maximum number of frames a track can be lost before being deleted
- `min_hits`: Minimum number of detections before a track is initialized
- `iou_threshold`: Intersection over Union threshold for matching detections to tracks

## How It Works

1. The system captures video frames from the specified source
2. Each frame is processed by the YOLO model to detect objects
3. Detections are passed to the SORT tracker to maintain object identities across frames
4. The results are visualized with bounding boxes, labels, and tracking IDs
5. The processed frame is displayed in real-time

## License

This project is open-source and available for personal and educational use.