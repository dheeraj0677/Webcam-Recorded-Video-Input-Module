"""
==============================================================================
VIDEO INPUT & MISSION TELEMETRY MODULE -- Stage 1 of the HAR Pipeline
==============================================================================

Author:  Dheeraj
Project: Space Experiment Monitoring (SIH Hackathon)
Role:    Stage 1 Video Ingestion, Quality Analytics & Frame Delivery

Core Capabilities:
  1. Universal Ingestion: Webcams, Local Video Files, and RTSP / IP Camera Streams.
  2. Zero-Latency Fresh Frame Mode: Eliminates backlog lag when downstream ML runs slowly.
  3. Space Lab Motion-Adaptive Sampling: Conserves compute & power during idle experiment phases.
  4. Real-Time Quality Analytics: Laplacian blur index, luminance analysis, and CLAHE enhancements.
  5. Thread-Safe Ring Buffering: Non-blocking I/O with per-frame microsecond timestamps.
  6. Standard & Rich Handoff: Simple (timestamp, frame) or advanced (frame, metadata).

==============================================================================
"""

import cv2
import numpy as np
import time
import threading
import argparse
from collections import deque
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Generator, Union


@dataclass
class FrameMetadata:
    """Rich telemetry metadata bundled with each captured frame."""
    frame_id: int
    timestamp: float
    fps: float
    latency_ms: float
    motion_detected: bool
    motion_score: float      # 0.0 to 100.0 scale
    blur_score: float        # Laplacian variance (higher = sharper)
    is_blurry: bool
    brightness: float        # Mean luminance (0.0 to 255.0)
    resolution: Tuple[int, int]
    source_type: str

    def to_dict(self) -> dict:
        """Convert metadata to dictionary for JSON/Telemetry serialization."""
        return asdict(self)


class VideoInputModule:
    """
    High-performance Stage 1 Video Ingestion and Telemetry Engine.
    """

    def __init__(
        self,
        source: Union[int, str] = 0,
        width: int = 640,
        height: int = 480,
        skip_frames: int = 1,
        buffer_size: int = 30,
        normalize: bool = False,
        zero_latency: bool = False,
        motion_adaptive: bool = False,
        detect_blur: bool = True,
        blur_threshold: float = 80.0,
        motion_threshold: float = 2.0,
    ):
        """
        Initialize the Video Input Module.

        Args:
            source: Webcam index (0, 1, ...), local video path ("video.mp4"),
                    or RTSP / HTTP streaming URL ("http://192.168.1.10:8080/video").
            width: Target frame width after resizing.
            height: Target frame height after resizing.
            skip_frames: Base frame skipping (process every Nth frame).
            buffer_size: Maximum rolling buffer capacity.
            normalize: Apply CLAHE contrast/lighting enhancement.
            zero_latency: Discard stale buffered frames to guarantee real-time feed.
            motion_adaptive: Dynamically throttle FPS when scene is idle.
            detect_blur: Compute Laplacian sharpness variance for quality gating.
            blur_threshold: Blur score cutoff below which frame is flagged as blurry.
            motion_threshold: Sensitivity threshold for motion detection (percentage).
        """
        self.source = source
        self.width = width
        self.height = height
        self.skip_frames = max(1, skip_frames)
        self.buffer_size = buffer_size
        self.normalize = normalize
        self.zero_latency = zero_latency
        self.motion_adaptive = motion_adaptive
        self.detect_blur = detect_blur
        self.blur_threshold = blur_threshold
        self.motion_threshold = motion_threshold

        # Source category classification
        if isinstance(source, int):
            self.source_type = "Webcam"
        elif isinstance(source, str) and (source.startswith("http://") or source.startswith("https://") or source.startswith("rtsp://")):
            self.source_type = "IP_Stream"
        else:
            self.source_type = "Video_File"

        # Internal State & Thread Safety
        self.cap = None
        self.frame_buffer = deque(maxlen=buffer_size)
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._frame_count = 0
        self._processed_count = 0
        self._fps = 0.0
        self._prev_gray = None

        # Pre-allocated CLAHE filter
        if self.normalize:
            self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        else:
            self._clahe = None

    # =========================================================================
    #  PUBLIC API
    # =========================================================================

    def start(self) -> bool:
        """
        Open video stream and start asynchronous background ingestion.
        """
        # Open source with OpenCV
        self.cap = cv2.VideoCapture(self.source)

        # Optimize RTSP buffer size if network stream
        if self.source_type == "IP_Stream":
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            print(f"[ERROR] Failed to open video source: {self.source}")
            return False

        native_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        native_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        native_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

        print(f"[INFO] Ingestion Source : {self.source_type} ({self.source})")
        print(f"[INFO] Native Stream    : {native_w}x{native_h} @ {native_fps:.1f} FPS")
        print(f"[INFO] Target Output    : {self.width}x{self.height}")
        print(f"[INFO] Zero Latency     : {'ENABLED' if self.zero_latency else 'OFF'}")
        print(f"[INFO] Motion Adaptive  : {'ENABLED' if self.motion_adaptive else 'OFF'}")
        print(f"[INFO] Blur Analytics   : {'ENABLED' if self.detect_blur else 'OFF'}")
        print(f"[INFO] Lighting CLAHE   : {'ENABLED' if self.normalize else 'OFF'}")

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        return True

    def read_frame(self, with_metadata: bool = False):
        """
        Retrieve the latest preprocessed frame from the buffer.
        """
        with self._lock:
            if len(self.frame_buffer) > 0:
                item = self.frame_buffer[-1]
                if self.zero_latency:
                    self.frame_buffer.clear()
                if with_metadata:
                    return item["frame"], item["meta"]
                return item["meta"].timestamp, item["frame"]
        return (None, None)

    def get_frames(self, with_metadata: bool = False) -> Generator:
        """
        Standard generator interface for downstream consumers.

        Args:
            with_metadata: If True, yields (frame, FrameMetadata).
                           If False (default), yields (timestamp, frame) for 100% backward compatibility.
        """
        last_frame_id = -1

        while self._running:
            item = None
            with self._lock:
                if len(self.frame_buffer) > 0:
                    latest = self.frame_buffer[-1]
                    if latest["meta"].frame_id != last_frame_id:
                        item = latest
                        last_frame_id = latest["meta"].frame_id
                        if self.zero_latency:
                            self.frame_buffer.clear()

            if item is not None:
                if with_metadata:
                    yield item["frame"], item["meta"]
                else:
                    yield item["meta"].timestamp, item["frame"]
            else:
                time.sleep(0.001)

    def stop(self):
        """Cleanly terminate background thread and release hardware resources."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        print("[INFO] Video Input Engine stopped. Resources released.")

    def get_fps(self) -> float:
        """Return real-time frame ingestion rate."""
        return self._fps

    def is_running(self) -> bool:
        """Return True if active."""
        return self._running

    # =========================================================================
    #  INTERNAL PROCESSING & ANALYTICS PIPELINE
    # =========================================================================

    def _capture_loop(self):
        """Asynchronous background ingestion and telemetry analysis worker."""
        frame_timestamps = deque(maxlen=30)
        idle_counter = 0

        while self._running:
            read_start = time.time()
            ret, raw_frame = self.cap.read()

            if not ret or raw_frame is None:
                if self.source_type == "Video_File":
                    print("[INFO] End of video stream reached.")
                else:
                    print("[WARN] Video stream disconnected or lost signal.")
                self._running = False
                break

            self._frame_count += 1
            capture_time = time.time()

            # --- Base Frame Skip Filter ---
            if self._frame_count % self.skip_frames != 0:
                continue

            # --- Step 1: Preprocessing (Resize & Color Conversion) ---
            resized_bgr = cv2.resize(raw_frame, (self.width, self.height))
            gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)

            # --- Step 2: Motion Energy Detection ---
            motion_score, motion_detected = self._detect_motion(gray)

            # --- Step 3: Space Lab Adaptive Power-Saving Throttling ---
            if self.motion_adaptive:
                if not motion_detected:
                    idle_counter += 1
                    # In idle mode, only process 1 out of every 6 frames (~5 FPS)
                    if idle_counter % 6 != 0:
                        continue
                else:
                    idle_counter = 0

            # --- Step 4: Quality & Blur Analytics ---
            blur_score, is_blurry = self._evaluate_blur(gray)
            mean_brightness = float(np.mean(gray))

            # --- Step 5: CLAHE Lighting Normalization ---
            if self.normalize and self._clahe is not None:
                lab = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2LAB)
                lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
                rgb_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            else:
                rgb_frame = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)

            # --- Calculate Latency and FPS ---
            frame_timestamps.append(capture_time)
            if len(frame_timestamps) >= 2:
                duration = frame_timestamps[-1] - frame_timestamps[0]
                if duration > 0:
                    self._fps = (len(frame_timestamps) - 1) / duration

            self._processed_count += 1
            latency_ms = (time.time() - read_start) * 1000.0

            # --- Construct Rich Frame Metadata ---
            meta = FrameMetadata(
                frame_id=self._processed_count,
                timestamp=capture_time,
                fps=round(self._fps, 1),
                latency_ms=round(latency_ms, 2),
                motion_detected=motion_detected,
                motion_score=round(motion_score, 2),
                blur_score=round(blur_score, 1),
                is_blurry=is_blurry,
                brightness=round(mean_brightness, 1),
                resolution=(self.width, self.height),
                source_type=self.source_type,
            )

            # --- Thread-Safe Buffer Push ---
            with self._lock:
                self.frame_buffer.append({"frame": rgb_frame, "meta": meta})

    def _detect_motion(self, current_gray: np.ndarray) -> Tuple[float, bool]:
        """Compute pixel-level motion energy via frame differencing."""
        if self._prev_gray is None:
            self._prev_gray = current_gray
            return 0.0, True

        frame_diff = cv2.absdiff(self._prev_gray, current_gray)
        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        non_zero_count = np.count_nonzero(thresh)
        total_pixels = current_gray.shape[0] * current_gray.shape[1]
        motion_score = (non_zero_count / total_pixels) * 100.0

        self._prev_gray = current_gray
        motion_detected = motion_score >= self.motion_threshold
        return motion_score, motion_detected

    def _evaluate_blur(self, gray: np.ndarray) -> Tuple[float, bool]:
        """Compute Laplacian variance to detect image blur/sharpness."""
        if not self.detect_blur:
            return 100.0, False

        score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        is_blurry = score < self.blur_threshold
        return score, is_blurry


# =============================================================================
#  CLI DEMO WITH HEADS-UP DISPLAY (HUD)
# =============================================================================

def draw_hud_overlay(frame_bgr: np.ndarray, meta: FrameMetadata) -> np.ndarray:
    """Renders a sleek mission telemetry HUD overlay onto the display frame."""
    hud = frame_bgr.copy()
    h, w, _ = hud.shape

    # Semi-transparent top telemetry banner
    cv2.rectangle(hud, (0, 0), (w, 55), (20, 24, 30), -1)
    cv2.addWeighted(hud, 0.75, frame_bgr, 0.25, 0, frame_bgr)

    # Top Status Text
    fps_color = (0, 255, 0) if meta.fps >= 20 else (0, 165, 255)
    cv2.putText(frame_bgr, f"FPS: {meta.fps:.1f}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)
    cv2.putText(frame_bgr, f"Latency: {meta.latency_ms:.1f}ms", (130, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    # Motion Status Pill
    motion_color = (0, 255, 100) if meta.motion_detected else (120, 120, 120)
    motion_text = f"MOTION: {'ACTIVE' if meta.motion_detected else 'IDLE'} ({meta.motion_score:.1f}%)"
    cv2.putText(frame_bgr, motion_text, (300, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, motion_color, 2)

    # Blur / Quality Pill
    blur_color = (0, 0, 255) if meta.is_blurry else (0, 255, 0)
    blur_text = f"SHARPNESS: {meta.blur_score:.0f} {'[BLUR]' if meta.is_blurry else '[OK]'}"
    cv2.putText(frame_bgr, blur_text, (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, blur_color, 1)

    # Frame Counter
    cv2.putText(frame_bgr, f"FRAME #{meta.frame_id}", (w - 140, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame_bgr, f"SRC: {meta.source_type}", (w - 140, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    return frame_bgr


def main():
    parser = argparse.ArgumentParser(description="Stage 1 -- Mission-Grade Video Input & Telemetry Engine")
    parser.add_argument("--source", type=str, default="0", help="Webcam index (0), video file path, or RTSP/HTTP URL")
    parser.add_argument("--width", type=int, default=640, help="Output width")
    parser.add_argument("--height", type=int, default=480, help="Output height")
    parser.add_argument("--skip", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--buffer", type=int, default=30, help="Rolling buffer capacity")
    parser.add_argument("--normalize", action="store_true", help="Enable CLAHE lighting normalization")
    parser.add_argument("--zero-latency", action="store_true", help="Enable zero-lag fresh frame mode")
    parser.add_argument("--motion-adaptive", action="store_true", help="Enable power-saving adaptive sampling")
    parser.add_argument("--no-blur-detect", action="store_true", help="Disable blur analytics")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    vim = VideoInputModule(
        source=source,
        width=args.width,
        height=args.height,
        skip_frames=args.skip,
        buffer_size=args.buffer,
        normalize=args.normalize,
        zero_latency=args.zero_latency,
        motion_adaptive=args.motion_adaptive,
        detect_blur=not args.no_blur_detect,
    )

    if not vim.start():
        return

    print("\n[INFO] Press 'q' in the display window to exit.\n")

    try:
        for frame, meta in vim.get_frames(with_metadata=True):
            display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            hud_frame = draw_hud_overlay(display_frame, meta)

            cv2.imshow("Space Mission HAR -- Stage 1 Telemetry HUD", hud_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        vim.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
