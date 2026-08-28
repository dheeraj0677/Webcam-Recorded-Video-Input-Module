# Webcam-Recorded-Video-Input-Module
### Stage 1: Mission-Grade Video Ingestion & Telemetry System for HAR Pipeline

**Author:** Dheeraj  
**Project:** Space Experiment Monitoring — Human Activity Recognition Pipeline  
**Hackathon:** SIH

---

## 📌 Overview

This is **Stage 1** of the 6-stage Space Experiment HAR pipeline. It provides an industry-grade video capture, real-time quality analytics, and telemetry engine supporting live webcams, pre-recorded video files, and wireless IP/RTSP streams.

```
[Stage 1: Video Ingestion & Telemetry] ➔ Object Detection ➔ Action Recognition ➔ Sequence Verification ➔ Mission Dashboard
```

---

## 🌟 Key Capabilities

1. **⚡ Zero-Latency Fresh Frame Mode:** Discards accumulated backlog when downstream deep learning models run slowly, ensuring the AI model always analyzes real-time live frames.
2. **🛰️ Space Lab Motion-Adaptive Sampling:** Conserves compute power and energy by automatically downsampling FPS when the experiment area is idle and instantly ramping up when movement is detected.
3. **🔍 Real-Time Blur & Quality Analytics:** Evaluates frame sharpness via Laplacian variance and calculates luminance to gate out blurry or under-exposed frames.
4. **📱 Universal Source Ingestion:** Supports USB webcams (`0`, `1`), local video files (`.mp4`, `.avi`), and wireless network streams (`rtsp://`, `http://192.168.x.x:8080/video`).
5. **🖥️ Live Mission Telemetry Web HUD (`dashboard.py`):** Sleek dark-mode browser dashboard showing live MJPEG feed, real-time FPS, ingestion latency, sharpness score, and motion indicators.

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Live Video Stream with Telemetry HUD

```bash
# Standard webcam with HUD overlay
python video_input.py

# Power-saving motion-adaptive mode + lighting normalization
python video_input.py --motion-adaptive --normalize

# Zero-latency mode for high-fps live feeds
python video_input.py --zero-latency

# Stream from an IP / Phone Camera (via Wi-Fi)
python video_input.py --source "http://192.168.1.10:8080/video"
```

### 3. Launch the Mission Web HUD Dashboard

```bash
python dashboard.py --source 0
```
Open **`http://localhost:5000`** in your browser to view the live dashboard and real-time telemetry gauges.

### 4. Run Automated Test Suite

```bash
python test_video_input.py
```

---

## ⚙️ Configuration Parameters

| Parameter | Default | Description |
|---|---|---|
| `--source` | `0` | Webcam index (`0`), local video file path, or RTSP/HTTP stream URL |
| `--width` | `640` | Output frame width |
| `--height` | `480` | Output frame height |
| `--skip` | `1` | Base frame skipping factor |
| `--buffer` | `30` | Rolling buffer size |
| `--normalize` | Off | Enable CLAHE contrast/lighting enhancement |
| `--zero-latency`| Off | Flush buffer on read to guarantee 0ms lag |
| `--motion-adaptive` | Off | Downsample FPS when workspace is idle |
| `--no-blur-detect` | Off | Disable Laplacian blur calculation |

---

## 🔌 How Downstream AI Stages Connect

### Pattern 1: Standard Simple Mode (100% Backward Compatible)
```python
from video_input import VideoInputModule

vim = VideoInputModule(source=0, width=640, height=480)
vim.start()

for timestamp, frame in vim.get_frames():
    # frame is RGB NumPy array: shape (480, 640, 3)
    results = detection_model.detect(frame)

vim.stop()
```

### Pattern 2: Rich Metadata Quality Gate
```python
from video_input import VideoInputModule

vim = VideoInputModule(source=0, detect_blur=True)
vim.start()

for frame, meta in vim.get_frames(with_metadata=True):
    if meta.is_blurry:
        continue  # Skip blurry frames to prevent false detections
    
    # Process crisp frames with full telemetry
    print(f"Frame #{meta.frame_id} | FPS: {meta.fps} | Motion: {meta.motion_score}%")

vim.stop()
```

See [`integration_example.py`](integration_example.py) for complete multi-case examples.

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────┐
│               UNIVERSAL INGESTION LAYER                │
│       Webcam (0)  │  Video File  │  RTSP / IP Stream   │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│           ASYNC BACKGROUND INGESTION THREAD            │
│   - Non-blocking frame capture (cv2.VideoCapture)      │
│   - Microsecond precision timestamping                 │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│               PREPROCESSING & ANALYTICS                │
│   1. Resizing & BGR ➔ RGB conversion                   │
│   2. Motion Differencing & Energy Score                │
│   3. Space Lab Adaptive FPS Throttling (Power Saver)   │
│   4. Laplacian Variance Blur & Sharpness Scoring       │
│   5. CLAHE Adaptive Contrast / Lighting Equalization   │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│            THREAD-SAFE TELEMETRY BUFFER                │
│   - Rolling deque with FrameMetadata encapsulation     │
│   - Zero-Latency auto-purge mode                       │
└─────────────┬────────────────────────────┬─────────────┘
              │                            │
              ▼                            ▼
┌───────────────────────────┐┌───────────────────────────┐
│     AI PIPELINE HANDOFF   ││     MISSION WEB HUD       │
│   get_frames() generator  ││   http://localhost:5000   │
└───────────────────────────┘└───────────────────────────┘
```

---

## 📂 Repository Structure

| File | Purpose |
|---|---|
| `video_input.py` | Core engine — `VideoInputModule` class & `FrameMetadata` dataclass |
| `dashboard.py` | Standalone Mission Telemetry Web HUD with live stream & gauges |
| `integration_example.py` | Integration patterns for downstream AI stages |
| `test_video_input.py` | Automated unit test suite (12/12 passing) |
| `requirements.txt` | Dependencies (`opencv-python`, `numpy`, `flask`) |
| `README.md` | System documentation |