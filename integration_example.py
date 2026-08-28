"""
==============================================================================
INTEGRATION EXAMPLES -- Connecting Downstream AI Models to Stage 1
==============================================================================

This file shows the 4 integration patterns for downstream teammates
(Object Detection, Action Recognition, and Sequence Check):

  1. Simple Mode: Standard (timestamp, frame) for 100% backward compatibility
  2. Mission Analytics Mode: (frame, FrameMetadata) with blur & motion telemetry
  3. Zero-Latency Mode: Real-time guarantee for heavy/slow AI models
  4. Space Lab Motion-Adaptive Mode: Power & compute saver for space stations

==============================================================================
"""

import time
from video_input import VideoInputModule


def example_1_simple_backward_compatible():
    """
    PATTERN 1: Standard Simple Mode
    Backward-compatible with original 3-line loop.
    """
    print("\n=== Pattern 1: Standard Simple Mode ===")
    vim = VideoInputModule(source=0, width=640, height=480)

    if not vim.start():
        return

    frame_count = 0
    for timestamp, frame in vim.get_frames():
        frame_count += 1
        # frame is RGB NumPy array: shape (480, 640, 3)
        # Udgeeth's Object Detection / YOLO model goes here:
        # results = detector.detect(frame)
        print(f"Received frame #{frame_count} @ {timestamp:.3f}s | Shape: {frame.shape}")

        if frame_count >= 30:
            break

    vim.stop()


def example_2_rich_metadata_telemetry():
    """
    PATTERN 2: Rich Metadata & Quality Gate
    Filter out blurry frames automatically before running heavy AI models.
    """
    print("\n=== Pattern 2: Rich Metadata & Quality Gate ===")
    vim = VideoInputModule(source=0, width=640, height=480, detect_blur=True)

    if not vim.start():
        return

    frame_count = 0
    for frame, meta in vim.get_frames(with_metadata=True):
        frame_count += 1

        # QUALITY GATE: Skip blurry frames to prevent false AI detections
        if meta.is_blurry:
            print(f"[SKIP] Frame #{meta.frame_id} flagged as BLURRY (Score: {meta.blur_score:.1f})")
            continue

        # AI INFERENCE: Process only high-quality frames
        print(f"[PROCESS] Frame #{meta.frame_id} | FPS: {meta.fps} | "
              f"Motion: {'ACTIVE' if meta.motion_detected else 'IDLE'} ({meta.motion_score:.1f}%) | "
              f"Sharpness: {meta.blur_score:.1f}")

        if frame_count >= 30:
            break

    vim.stop()


def example_3_zero_latency_mode():
    """
    PATTERN 3: Zero-Latency Mode for Slow AI Inference
    Guarantees no lag backlog even when the AI model runs at 5 FPS.
    """
    print("\n=== Pattern 3: Zero-Latency Mode for Heavy Models ===")
    vim = VideoInputModule(source=0, width=640, height=480, zero_latency=True)

    if not vim.start():
        return

    frame_count = 0
    for timestamp, frame in vim.get_frames():
        frame_count += 1

        # Simulate a heavy deep learning model taking 150ms per frame (~6 FPS)
        time.sleep(0.15)

        # In Zero-Latency mode, the next frame will be the FRESH live camera frame,
        # never an old queued frame from 2 seconds ago.
        print(f"Processed frame #{frame_count} (Zero-Lag Guaranteed)")

        if frame_count >= 15:
            break

    vim.stop()


def example_4_space_lab_motion_adaptive():
    """
    PATTERN 4: Space Lab Motion-Adaptive Sampling
    Automatically downsamples when experiment workspace is idle.
    """
    print("\n=== Pattern 4: Space Lab Motion-Adaptive Sampling ===")
    vim = VideoInputModule(source=0, width=640, height=480, motion_adaptive=True)

    if not vim.start():
        return

    frame_count = 0
    for frame, meta in vim.get_frames(with_metadata=True):
        frame_count += 1
        state = "ACTIVE EXPERIMENT" if meta.motion_detected else "IDLE (POWER SAVING)"
        print(f"Frame #{meta.frame_id} | State: {state} | Energy: {meta.motion_score:.1f}%")

        if frame_count >= 30:
            break

    vim.stop()


if __name__ == "__main__":
    print("Choose an integration pattern to test:")
    print("  1 - Standard Simple Mode (Backward Compatible)")
    print("  2 - Rich Metadata & Quality Gate (Skip Blurry Frames)")
    print("  3 - Zero-Latency Mode (For Slow AI Inference)")
    print("  4 - Space Lab Motion-Adaptive Mode (Power Saver)")

    choice = input("\nEnter choice (1-4): ").strip()
    if choice == "1":
        example_1_simple_backward_compatible()
    elif choice == "2":
        example_2_rich_metadata_telemetry()
    elif choice == "3":
        example_3_zero_latency_mode()
    elif choice == "4":
        example_4_space_lab_motion_adaptive()
    else:
        example_1_simple_backward_compatible()
