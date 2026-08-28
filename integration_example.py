"""
==============================================================================
INTEGRATION EXAMPLE — How Teammates Use the Video Input Module
==============================================================================

This file shows how Udgeeth (or any downstream module) connects to
Dheeraj's VideoInputModule to receive preprocessed frames.

The interface is simple:
  1. Import VideoInputModule
  2. Create an instance with your source (webcam or file)
  3. Call start()
  4. Loop over get_frames() — each iteration gives you a (timestamp, frame)
  5. Call stop() when done

That's it. Your detection model goes inside the loop.
==============================================================================
"""

from video_input import VideoInputModule


def example_1_basic_webcam():
    """
    BASIC EXAMPLE: Read from webcam, print frame info.
    This is the simplest possible usage.
    """
    print("=== Example 1: Basic Webcam ===\n")

    # Create the module — source=0 means default webcam
    vim = VideoInputModule(source=0, width=640, height=480)

    # Start capturing
    if not vim.start():
        print("Could not start webcam!")
        return

    # Read frames — this is where Udgeeth's detection would go
    frame_count = 0
    for timestamp, frame in vim.get_frames():
        frame_count += 1

        # =====================================================
        # UDGEETH: Put your detection model call here!
        #
        #   detections = your_model.detect(frame)
        #   for obj in detections:
        #       print(f"Detected: {obj.label} ({obj.confidence:.2f})")
        #
        # The 'frame' is already:
        #   - Resized to 640x480
        #   - Converted to RGB
        #   - A NumPy array, shape (480, 640, 3), dtype uint8
        # =====================================================

        print(f"Frame {frame_count}: shape={frame.shape}, timestamp={timestamp:.3f}")

        # Stop after 100 frames for this demo
        if frame_count >= 100:
            break

    vim.stop()
    print(f"\nProcessed {frame_count} frames.\n")


def example_2_video_file():
    """
    VIDEO FILE EXAMPLE: Read from a recorded .mp4 file.
    Change the path to your actual video file.
    """
    print("=== Example 2: Video File ===\n")

    vim = VideoInputModule(
        source="experiment_recording.mp4",  # <-- change this to your file
        width=640,
        height=480,
        skip_frames=2,     # process every 2nd frame (halves workload)
        normalize=True,    # brighten dark footage
    )

    if not vim.start():
        print("Could not open video file! Check the path.")
        return

    for timestamp, frame in vim.get_frames():
        # Your processing here...
        print(f"Frame shape: {frame.shape}, FPS: {vim.get_fps():.1f}")

    vim.stop()
    print("Video file processing complete.\n")


def example_3_detection_integration():
    """
    FULL INTEGRATION SKELETON — This is how the complete pipeline connection
    between Dheeraj (Stage 1) and Udgeeth (Stage 2) would look.

    Udgeeth: copy this pattern and replace the dummy function with your
    actual YOLO/detection model.
    """
    print("=== Example 3: Detection Integration Skeleton ===\n")

    # --- Dheeraj's module (Stage 1) ---
    vim = VideoInputModule(source=0, width=640, height=480)

    if not vim.start():
        return

    frame_count = 0

    for timestamp, frame in vim.get_frames():
        frame_count += 1

        # --- Udgeeth's detection (Stage 2) ---
        # Replace this dummy function with your real model:
        #
        #   from detection_module import ObjectDetector
        #   detector = ObjectDetector("yolov8n.pt")
        #   results = detector.detect(frame)
        #
        detections = dummy_detect(frame)

        # --- Arpit's action recognition (Stage 3) ---
        # action = recognize_action(frame, detections)

        # --- Shalini's sequence check (Stage 4) ---
        # is_correct = check_sequence(action)

        # --- Varshitha's result (Stage 5) ---
        # result = classify_step(is_correct)

        # Print progress
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames | FPS: {vim.get_fps():.1f}")

        if frame_count >= 150:
            break

    vim.stop()
    print(f"\nDone. Total frames: {frame_count}\n")


def dummy_detect(frame):
    """
    Placeholder detection function.
    Udgeeth will replace this with the real object detection model.

    Args:
        frame: RGB numpy array, shape (H, W, 3)

    Returns:
        list: Detected objects (empty for now)
    """
    return []


# =============================================================================
#  Run any of the examples
# =============================================================================
if __name__ == "__main__":
    print("Choose an example to run:")
    print("  1 — Basic webcam capture")
    print("  2 — Video file processing")
    print("  3 — Detection integration skeleton")
    print()

    choice = input("Enter 1, 2, or 3: ").strip()

    if choice == "1":
        example_1_basic_webcam()
    elif choice == "2":
        example_2_video_file()
    elif choice == "3":
        example_3_detection_integration()
    else:
        print("Invalid choice. Running Example 1 by default.")
        example_1_basic_webcam()
