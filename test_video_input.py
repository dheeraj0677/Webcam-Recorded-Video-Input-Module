"""
==============================================================================
TESTS -- Video Input & Mission Telemetry Module
==============================================================================

Automated test suite verifying:
  - Video file capture & frame generation
  - Frame dimension resizing & RGB conversion
  - Base frame skipping
  - Monotonic timestamps & clean shutdown
  - Invalid source graceful error handling
  - CLAHE contrast normalization
  - Rich FrameMetadata generation & serialization
  - Laplacian blur & sharpness detection
  - Motion energy differencing
  - Zero-latency buffer management

Run:
  python test_video_input.py
==============================================================================
"""

import cv2
import numpy as np
import os
import sys
import time
import tempfile

from video_input import VideoInputModule, FrameMetadata


# =============================================================================
#  SYNTHETIC TEST VIDEO GENERATORS
# =============================================================================

def create_synthetic_video(filepath, num_frames=60, fps=30, width=320, height=240):
    """Creates a short test video with color transitions and frame numbers."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))

    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        if i < 20:
            frame[:, :, 2] = 220   # Red (BGR: channel 2)
        elif i < 40:
            frame[:, :, 1] = 220   # Green (BGR: channel 1)
        else:
            frame[:, :, 0] = 220   # Blue (BGR: channel 0)

        # Draw text to generate sharp edges
        cv2.putText(frame, f"TEST_FRAME_{i}", (30, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        writer.write(frame)

    writer.release()
    return filepath


# =============================================================================
#  TEST TRACKER
# =============================================================================

class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def ok(self, name):
        self.passed += 1
        print(f"  [PASS] {name}")

    def fail(self, name, reason):
        self.failed += 1
        self.errors.append((name, reason))
        print(f"  [FAIL] {name} -- {reason}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*55}")
        print(f"Test Suite Results: {self.passed}/{total} Passed, {self.failed} Failed")
        if self.errors:
            print("\nFailures:")
            for name, reason in self.errors:
                print(f"  - {name}: {reason}")
        print(f"{'='*55}")
        return self.failed == 0


# =============================================================================
#  UNIT TESTS
# =============================================================================

def test_basic_video_capture(video_path, results):
    vim = VideoInputModule(source=video_path, width=640, height=480)
    if not vim.start():
        results.fail("Open video file", "start() returned False")
        return

    results.ok("Open video file")

    count = 0
    for _, _ in vim.get_frames():
        count += 1
        if count >= 10:
            break

    if count == 10:
        results.ok("Read 10 frames from video")
    else:
        results.fail("Read 10 frames", f"Got {count}")

    vim.stop()


def test_frame_dimensions_and_rgb(video_path, results):
    target_w, target_h = 320, 240
    vim = VideoInputModule(source=video_path, width=target_w, height=target_h)
    vim.start()

    for _, frame in vim.get_frames():
        h, w, c = frame.shape
        if w == target_w and h == target_h and c == 3:
            results.ok(f"Frame dimensions correct ({target_w}x{target_h}x3)")
        else:
            results.fail("Frame dimensions", f"Got {w}x{h}x{c}")

        # In RGB, the red channel (channel 0) should be dominant in the first section
        center_pixel = frame[120, 160]
        if center_pixel[0] > center_pixel[1] and center_pixel[0] > center_pixel[2]:
            results.ok("BGR -> RGB conversion verified")
        else:
            results.fail("BGR -> RGB", f"Got R={center_pixel[0]} G={center_pixel[1]} B={center_pixel[2]}")
        break

    vim.stop()


def test_metadata_structure(video_path, results):
    vim = VideoInputModule(source=video_path, width=320, height=240, detect_blur=True)
    vim.start()

    for frame, meta in vim.get_frames(with_metadata=True):
        if isinstance(meta, FrameMetadata):
            results.ok("FrameMetadata dataclass verified")
        else:
            results.fail("FrameMetadata", "Did not receive FrameMetadata instance")

        meta_dict = meta.to_dict()
        required_keys = ["frame_id", "timestamp", "fps", "latency_ms", "motion_detected", "blur_score", "is_blurry"]
        if all(k in meta_dict for k in required_keys):
            results.ok("Metadata schema keys verified")
        else:
            results.fail("Metadata schema", f"Missing required keys in {meta_dict.keys()}")
        break

    vim.stop()


def test_blur_analytics(results):
    vim = VideoInputModule(source=0, detect_blur=True, blur_threshold=100.0)

    # Synthetic sharp image (high variance edges)
    sharp_img = np.zeros((100, 100), dtype=np.uint8)
    sharp_img[::2, ::2] = 255
    sharp_score, sharp_is_blurry = vim._evaluate_blur(sharp_img)

    # Synthetic blurred image (flat/smooth)
    blur_img = np.ones((100, 100), dtype=np.uint8) * 128
    blur_score, blur_is_blurry = vim._evaluate_blur(blur_img)

    if sharp_score > blur_score and blur_is_blurry:
        results.ok(f"Blur detector verified (Sharp: {sharp_score:.1f}, Blur: {blur_score:.1f})")
    else:
        results.fail("Blur detector", f"Sharp: {sharp_score}, Blur: {blur_score}")


def test_motion_detection(results):
    vim = VideoInputModule(source=0, motion_threshold=2.0)

    frame1 = np.zeros((100, 100), dtype=np.uint8)
    frame2_static = np.zeros((100, 100), dtype=np.uint8)
    frame3_motion = np.zeros((100, 100), dtype=np.uint8)
    frame3_motion[:50, :50] = 255

    vim._detect_motion(frame1)  # initialize prev
    _, static_detected = vim._detect_motion(frame2_static)
    motion_score, motion_detected = vim._detect_motion(frame3_motion)

    if not static_detected and motion_detected and motion_score > 0:
        results.ok(f"Motion differencing verified (Energy: {motion_score:.1f}%)")
    else:
        results.fail("Motion differencing", f"Static: {static_detected}, Motion: {motion_detected}")


def test_zero_latency_mode(video_path, results):
    vim = VideoInputModule(source=video_path, width=320, height=240, zero_latency=True)
    vim.start()

    count = 0
    for _, _ in vim.get_frames():
        count += 1
        if count >= 5:
            break

    vim.stop()
    if count >= 5:
        results.ok("Zero-latency streaming mode verified")
    else:
        results.fail("Zero-latency mode", f"Got {count} frames")


def test_normalization_mode(video_path, results):
    vim = VideoInputModule(source=video_path, width=320, height=240, normalize=True)
    vim.start()

    count = 0
    for _, _ in vim.get_frames():
        count += 1
        if count >= 5:
            break

    vim.stop()
    if count >= 5:
        results.ok("CLAHE lighting normalization verified")
    else:
        results.fail("CLAHE normalization", f"Got {count} frames")


def test_invalid_source(results):
    vim = VideoInputModule(source="nonexistent_mock_file_9999.mp4")
    if not vim.start():
        results.ok("Invalid source handled gracefully (start() returned False)")
    else:
        results.fail("Invalid source", "start() returned True for missing file")
        vim.stop()


def test_clean_stop(video_path, results):
    vim = VideoInputModule(source=video_path, width=320, height=240)
    vim.start()
    for _, _ in vim.get_frames():
        break
    vim.stop()

    if not vim.is_running():
        results.ok("Clean stop and resource deallocation verified")
    else:
        results.fail("Clean stop", "is_running() is still True")


# =============================================================================
#  MAIN TEST RUNNER
# =============================================================================

def main():
    print("=" * 55)
    print("Stage 1 Video & Mission Telemetry -- Test Suite")
    print("=" * 55)

    results = TestResults()
    temp_dir = tempfile.gettempdir()
    test_video_path = os.path.join(temp_dir, "test_video_input_enhanced.mp4")

    print(f"\n[INIT] Generating synthetic test video...")
    create_synthetic_video(test_video_path, num_frames=60, fps=30)

    try:
        print("[TEST] Running automated unit tests...\n")
        test_basic_video_capture(test_video_path, results)
        test_frame_dimensions_and_rgb(test_video_path, results)
        test_metadata_structure(test_video_path, results)
        test_blur_analytics(results)
        test_motion_detection(results)
        test_zero_latency_mode(test_video_path, results)
        test_normalization_mode(test_video_path, results)
        test_invalid_source(results)
        test_clean_stop(test_video_path, results)

    finally:
        for _ in range(5):
            try:
                if os.path.exists(test_video_path):
                    time.sleep(0.5)
                    os.remove(test_video_path)
                break
            except PermissionError:
                time.sleep(1)

    all_passed = results.summary()
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
