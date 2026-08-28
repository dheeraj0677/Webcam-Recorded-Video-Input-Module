# Webcam-Recorded-Video-Input-Module
### Stage 1: Video Capture & Frame Extraction Module for HAR Pipeline

**Author:** Dheeraj  
**Project:** ISRO Space Experiment Monitoring — Human Activity Recognition Pipeline  
**Hackathon:** SIH

---

## 📌 What This Module Does

This is **Stage 1** of the 6-stage HAR pipeline. It captures video from a webcam or a recorded file, preprocesses each frame (resizing, BGR→RGB conversion, optional CLAHE brightness normalization), and delivers clean frames via a thread-safe generator to the downstream detection module.

```
[Video Input Module] → Object Detection → Action Recognition → Sequence Check → Verification → Dashboard
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Live Demo (Webcam)

```bash
python video_input.py
```

This opens your webcam and shows the preprocessed feed with a real-time FPS overlay. Press **'q'** to quit.

### 3. Run with a Video File

```bash
python video_input.py --source path/to/video.mp4
```

### 4. Run Tests

```bash
python test_video_input.py
```

---

## ⚙️ Configuration Options

| Argument | Default | Description |
|---|---|---|
| `--source` | `0` (webcam) | Webcam index (`0`, `1`, ...) or path to video file (`.mp4`, etc.) |
| `--width` | `640` | Output frame width |
| `--height` | `480` | Output frame height |
| `--skip` | `1` | Process every Nth frame (e.g. 2 = half FPS, 3 = third) to reduce ML load |
| `--buffer` | `30` | Max frames to hold in the rolling buffer |
| `--normalize` | Off | Enable CLAHE contrast/brightness normalization for low-light lab conditions |

---

## 🔌 How Teammates Use This Module

Udgeeth (or any downstream module) imports the `VideoInputModule` and loops over `get_frames()`:

```python
from video_input import VideoInputModule

# 1. Initialize — choose webcam or video file
vim = VideoInputModule(source=0, width=640, height=480)

# 2. Start threaded background capture
vim.start()

# 3. Process frames — detection model goes here
for timestamp, frame in vim.get_frames():
    # frame is a NumPy array: shape (480, 640, 3), RGB, uint8
    results = your_detection_model(frame)
    print(f"Time: {timestamp:.3f}, Detections: {results}")

# 4. Release resources cleanly
vim.stop()
```

See [`integration_example.py`](integration_example.py) for complete multi-case examples.

---

## 🏗️ Architecture

```
┌─────────────────────────────┐
│      INPUT SOURCE SELECT     │
│   (Webcam OR Recorded Video) │
└───────────────┬──────────────┘
                │
┌───────────────▼──────────────┐
│       VIDEO CAPTURE           │
│   cv2.VideoCapture(source)    │
│   Runs in background thread   │
└───────────────┬──────────────┘
                │
┌───────────────▼──────────────┐
│     FRAME EXTRACTION          │
│   Skip every Nth frame        │
│   Control processing rate     │
└───────────────┬──────────────┘
                │
┌───────────────▼──────────────┐
│      PREPROCESSING            │
│   1. Resize (640×480)         │
│   2. BGR → RGB conversion     │
│   3. CLAHE normalization      │
└───────────────┬──────────────┘
                │
┌───────────────▼──────────────┐
│     FRAME BUFFER (deque)      │
│   Thread-safe rolling buffer  │
│   Timestamped frames          │
└───────────────┬──────────────┘
                │
┌───────────────▼──────────────┐
│   OUTPUT: get_frames()        │
│   Generator yielding          │
│   (timestamp, frame) tuples   │
└──────────────────────────────┘
```

---

## 📂 Repository Structure

| File | Purpose |
|---|---|
| `video_input.py` | Core module — `VideoInputModule` class with threaded capture and preprocessing |
| `integration_example.py` | Standalone script showing teammates how to connect their stages |
| `test_video_input.py` | Automated unit test suite (10/10 verified passing) |
| `requirements.txt` | Minimal dependencies (`opencv-python`, `numpy`) |
| `README.md` | Documentation and architecture guide |

---

## 📖 API Reference

### `VideoInputModule(source, width, height, skip_frames, buffer_size, normalize)`

| Method | Returns | Description |
|---|---|---|
| `start()` | `bool` | Opens video source, starts background capture thread |
| `read_frame()` | `(timestamp, frame)` | Get the latest frame from the buffer |
| `get_frames()` | Generator | Yields `(timestamp, frame)` tuples continuously |
| `stop()` | None | Stops capture, releases camera/file handle cleanly |
| `get_fps()` | `float` | Current measured FPS |
| `is_running()` | `bool` | Whether the capture loop is active |