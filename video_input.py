"""
==============================================================================
VIDEO INPUT MODULE — Stage 1 of the HAR Pipeline
==============================================================================

Author:  Dheeraj
Project: ISRO Space Experiment Monitoring (SIH Hackathon)
Role:    Captures video from webcam or file, preprocesses frames,
         and delivers them to the downstream detection stage (Udgeeth).

Pipeline Position:
  [YOU] → Object+Activity Detection → Action Recognition →
  Sequence Check → Correct/Wrong → Dashboard

Usage:
  # As a standalone demo (shows live webcam feed):
  python video_input.py

  # With a recorded video file:
  python video_input.py --source path/to/video.mp4

  # From another module (Udgeeth's detection code):
  from video_input import VideoInputModule
  vim = VideoInputModule(source=0)
  vim.start()
  for timestamp, frame in vim.get_frames():
      # process frame here
      pass
  vim.stop()
==============================================================================
"""

import cv2
import numpy as np
import time
import threading
import argparse
from collections import deque


class VideoInputModule:
    """
    Stage 1 of the HAR pipeline — captures video, preprocesses frames,
    and exposes them through a generator for downstream consumption.

    This class handles:
    1. Source selection (webcam or video file)
    2. Threaded frame capture (non-blocking)
    3. Frame preprocessing (resize, color conversion, normalization)
    4. Buffered output with timestamps
    """

    def __init__(
        self,
        source=0,
        width=640,
        height=480,
        skip_frames=1,
        buffer_size=30,
        normalize=False,
    ):
        """
        Initialize the Video Input Module.

        Args:
            source (int or str):
                - 0 (or other int): Use webcam at that index.
                  0 = default camera, 1 = second camera, etc.
                - "path/to/video.mp4": Use a recorded video file.

            width (int): Target frame width after resize. Default 640.
            height (int): Target frame height after resize. Default 480.

            skip_frames (int): Process every Nth frame. Default 1 (every frame).
                - Set to 2 to halve the frame rate, 3 for one-third, etc.
                - Useful to reduce load on the downstream detection model.

            buffer_size (int): Max frames to hold in the rolling buffer.
                - If the buffer is full, oldest frames are dropped.
                - Default 30 (about 1 second at 30 FPS).

            normalize (bool): Whether to apply CLAHE brightness normalization.
                - Useful for low-light lab/experiment conditions.
                - Default False (skip normalization for speed).
        """
        # --- Configuration ---
        self.source = source
        self.width = width
        self.height = height
        self.skip_frames = max(1, skip_frames)  # at least 1 (every frame)
        self.buffer_size = buffer_size
        self.normalize = normalize

        # --- Internal State ---
        self.cap = None                         # OpenCV VideoCapture object
        self.frame_buffer = deque(maxlen=buffer_size)  # rolling frame buffer
        self._running = False                   # flag to control the capture thread
        self._thread = None                     # background capture thread
        self._lock = threading.Lock()           # thread safety for the buffer
        self._frame_count = 0                   # total frames read from source
        self._fps = 0.0                         # calculated FPS

        # --- CLAHE for brightness normalization (created once, reused) ---
        if self.normalize:
            self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        else:
            self._clahe = None

    # =========================================================================
    #  PUBLIC METHODS — These are what you (and your teammates) call
    # =========================================================================

    def start(self):
        """
        Open the video source and start capturing frames in the background.

        Returns:
            bool: True if the source was opened successfully, False otherwise.
        """
        # Open the video source
        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            print(f"[ERROR] Could not open video source: {self.source}")
            return False

        # Print source info
        src_type = "Webcam" if isinstance(self.source, int) else "Video File"
        src_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        src_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"[INFO] Source: {src_type} ({self.source})")
        print(f"[INFO] Native resolution: {src_width}x{src_height} @ {src_fps:.1f} FPS")
        print(f"[INFO] Output resolution: {self.width}x{self.height}")
        print(f"[INFO] Frame skip: every {self.skip_frames} frame(s)")
        print(f"[INFO] Normalization: {'ON' if self.normalize else 'OFF'}")
        print(f"[INFO] Buffer size: {self.buffer_size} frames")
        print()

        # Start the background capture thread
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        print("[INFO] Capture started. Frames are being read in the background.")
        return True

    def read_frame(self):
        """
        Get the most recent preprocessed frame from the buffer.

        Returns:
            tuple: (timestamp, frame) if available, or (None, None) if buffer is empty.
                - timestamp (float): Time the frame was captured (time.time()).
                - frame (np.ndarray): Preprocessed frame as a NumPy array (RGB, uint8).
        """
        with self._lock:
            if len(self.frame_buffer) > 0:
                return self.frame_buffer[-1]  # return the latest frame
        return None, None

    def get_frames(self):
        """
        Generator that yields preprocessed frames one at a time.

        THIS IS THE MAIN INTERFACE for downstream modules.
        Udgeeth's detection code should call this in a for-loop:

            vim = VideoInputModule(source=0)
            vim.start()
            for timestamp, frame in vim.get_frames():
                detections = your_model.detect(frame)
                ...

        Yields:
            tuple: (timestamp, frame)
                - timestamp (float): Time the frame was captured.
                - frame (np.ndarray): Preprocessed RGB frame as a NumPy array,
                  shape (height, width, 3), dtype uint8.
        """
        last_timestamp = None

        while self._running:
            with self._lock:
                if len(self.frame_buffer) > 0:
                    timestamp, frame = self.frame_buffer[-1]

                    # Only yield if this is a NEW frame (avoid duplicates)
                    if timestamp != last_timestamp:
                        last_timestamp = timestamp
                        yield timestamp, frame
                else:
                    # Buffer empty — wait a tiny bit before checking again
                    pass

            # Small sleep to prevent busy-waiting and hogging the CPU
            time.sleep(0.001)

    def stop(self):
        """
        Stop capturing and release all resources.
        Always call this when you're done — cleans up the webcam/file handle.
        """
        self._running = False

        # Wait for the capture thread to finish
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        # Release the video source
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        print("[INFO] Video capture stopped and resources released.")

    def get_fps(self):
        """Return the current measured FPS of the capture loop."""
        return self._fps

    def is_running(self):
        """Check if the capture loop is still active."""
        return self._running

    # =========================================================================
    #  PRIVATE METHODS — Internal machinery, you don't call these directly
    # =========================================================================

    def _capture_loop(self):
        """
        Background thread: continuously reads frames from the video source,
        preprocesses them, and pushes them into the frame buffer.

        This runs in a separate thread so the main thread (where detection
        happens) is never blocked waiting for the camera.
        """
        frame_times = deque(maxlen=30)  # for FPS calculation

        while self._running:
            ret, raw_frame = self.cap.read()

            # --- Handle end-of-video or camera disconnect ---
            if not ret or raw_frame is None:
                if isinstance(self.source, int):
                    # Webcam disconnected — stop
                    print("[WARN] Webcam disconnected or frame read failed.")
                else:
                    # End of video file — stop
                    print("[INFO] End of video file reached.")
                self._running = False
                break

            self._frame_count += 1

            # --- Frame skipping: only process every Nth frame ---
            if self._frame_count % self.skip_frames != 0:
                continue

            # --- Preprocess the frame ---
            processed_frame = self._preprocess(raw_frame)

            # --- Timestamp this frame ---
            timestamp = time.time()

            # --- Push into the thread-safe buffer ---
            with self._lock:
                self.frame_buffer.append((timestamp, processed_frame))

            # --- Calculate FPS ---
            frame_times.append(timestamp)
            if len(frame_times) >= 2:
                elapsed = frame_times[-1] - frame_times[0]
                if elapsed > 0:
                    self._fps = (len(frame_times) - 1) / elapsed

    def _preprocess(self, frame):
        """
        Apply the preprocessing pipeline to a raw frame.

        Steps:
        1. Resize to target dimensions (width x height)
        2. Convert BGR → RGB (OpenCV reads as BGR, but most ML models want RGB)
        3. Optionally normalize brightness using CLAHE

        Args:
            frame (np.ndarray): Raw BGR frame from OpenCV.

        Returns:
            np.ndarray: Preprocessed RGB frame, shape (height, width, 3).
        """
        # Step 1: Resize to configured resolution
        frame = cv2.resize(frame, (self.width, self.height))

        # Step 2: Convert BGR to RGB
        #   OpenCV captures in BGR, but most detection/ML models expect RGB.
        #   This conversion happens here so Udgeeth's code doesn't have to.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Step 3: Optional brightness normalization (CLAHE)
        #   Useful when the experiment is in a dimly-lit lab.
        #   CLAHE = Contrast Limited Adaptive Histogram Equalization
        if self.normalize and self._clahe is not None:
            # CLAHE works on single channels, so convert to LAB color space,
            # equalize the L (lightness) channel, then convert back.
            lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return frame


# =============================================================================
#  DEMO — Run this file directly to test with your webcam
# =============================================================================

def main():
    """
    Demo: Opens the webcam (or a video file), shows the live preprocessed
    feed in a window, and prints FPS to the console.

    Usage:
        python video_input.py                          # webcam
        python video_input.py --source video.mp4       # video file
        python video_input.py --width 320 --height 240 # lower resolution
        python video_input.py --normalize               # brightness normalization
        python video_input.py --skip 3                  # process every 3rd frame
    """
    # --- Parse command-line arguments ---
    parser = argparse.ArgumentParser(
        description="Stage 1 — Video Input Module for HAR Pipeline"
    )
    parser.add_argument(
        "--source", type=str, default="0",
        help="Video source: 0 for webcam, or path to video file (default: 0)"
    )
    parser.add_argument("--width", type=int, default=640, help="Frame width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Frame height (default: 480)")
    parser.add_argument("--skip", type=int, default=1, help="Process every Nth frame (default: 1)")
    parser.add_argument("--buffer", type=int, default=30, help="Buffer size (default: 30)")
    parser.add_argument("--normalize", action="store_true", help="Enable brightness normalization")
    args = parser.parse_args()

    # Convert source: if it's a digit string, treat as webcam index
    source = int(args.source) if args.source.isdigit() else args.source

    # --- Create and start the module ---
    vim = VideoInputModule(
        source=source,
        width=args.width,
        height=args.height,
        skip_frames=args.skip,
        buffer_size=args.buffer,
        normalize=args.normalize,
    )

    if not vim.start():
        print("[ERROR] Failed to start. Check your webcam or file path.")
        return

    print("[INFO] Press 'q' in the video window to quit.\n")

    frame_count = 0

    try:
        for timestamp, frame in vim.get_frames():
            frame_count += 1

            # --- Convert RGB back to BGR for OpenCV display ---
            # (Our module outputs RGB for ML models, but cv2.imshow expects BGR)
            display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # --- Draw FPS overlay on the display frame ---
            fps_text = f"FPS: {vim.get_fps():.1f}"
            cv2.putText(
                display_frame, fps_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
            )

            # --- Draw frame count ---
            count_text = f"Frame: {frame_count}"
            cv2.putText(
                display_frame, count_text, (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

            # --- Draw resolution info ---
            res_text = f"Size: {frame.shape[1]}x{frame.shape[0]}"
            cv2.putText(
                display_frame, res_text, (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

            # --- Show the frame ---
            cv2.imshow("HAR Pipeline - Stage 1 (Dheeraj)", display_frame)

            # --- Print FPS to console every 30 frames ---
            if frame_count % 30 == 0:
                print(f"[FPS] {vim.get_fps():.1f} | Frames processed: {frame_count}")

            # --- Check for 'q' key to quit ---
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[INFO] 'q' pressed — stopping...")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C pressed — stopping...")

    finally:
        vim.stop()
        cv2.destroyAllWindows()
        print(f"[INFO] Total frames processed: {frame_count}")


if __name__ == "__main__":
    main()
