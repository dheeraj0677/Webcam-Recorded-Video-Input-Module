"""
==============================================================================
MISSION TELEMETRY WEB HUD DASHBOARD -- Stage 1 Video Ingestion
==============================================================================

Author:  Dheeraj
Project: Space Experiment Monitoring (SIH Hackathon)
Role:    Real-Time Browser HUD with Telemetry Dials & Live Stream

Run:
  python dashboard.py
  Then open in browser: http://localhost:5000
==============================================================================
"""

import cv2
import json
import time
import argparse
from flask import Flask, Response, render_template_string, jsonify
from video_input import VideoInputModule, FrameMetadata

app = Flask(__name__)

# Global module instance and latest telemetry store
vim_instance = None
latest_telemetry = {}

HTML_DASHBOARD = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stage 1 -- Mission Telemetry & Video HUD</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0a0d14;
            --bg-card: rgba(18, 24, 38, 0.75);
            --border-card: rgba(255, 255, 255, 0.08);
            --accent-cyan: #00f2fe;
            --accent-blue: #4facfe;
            --accent-green: #00e676;
            --accent-orange: #ff9100;
            --accent-red: #ff3d00;
            --text-primary: #f0f4f8;
            --text-muted: #8a99ad;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            background-color: var(--bg-primary);
            background-image: radial-gradient(circle at 10% 20%, rgba(0, 242, 254, 0.04) 0%, transparent 40%),
                              radial-gradient(circle at 90% 80%, rgba(79, 172, 254, 0.04) 0%, transparent 40%);
            color: var(--text-primary);
            font-family: 'Inter', sans-serif;
            min-height: 100vh;
            padding: 24px;
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--border-card);
            margin-bottom: 24px;
        }

        .header-title h1 {
            font-size: 1.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
        }

        .header-title p {
            color: var(--text-muted);
            font-size: 0.85rem;
            margin-top: 4px;
        }

        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: rgba(0, 230, 118, 0.1);
            color: var(--accent-green);
            padding: 6px 14px;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
            border: 1px solid rgba(0, 230, 118, 0.2);
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: var(--accent-green);
            box-shadow: 0 0 10px var(--accent-green);
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { transform: scale(0.95); opacity: 0.8; }
            50% { transform: scale(1.2); opacity: 1; }
            100% { transform: scale(0.95); opacity: 0.8; }
        }

        .grid-container {
            display: grid;
            grid-template-columns: 1fr 340px;
            gap: 24px;
        }

        @media (max-width: 960px) {
            .grid-container {
                grid-template-columns: 1fr;
            }
        }

        .video-card {
            background: var(--bg-card);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-card);
            border-radius: 16px;
            padding: 16px;
            position: relative;
            overflow: hidden;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
        }

        .video-wrapper {
            position: relative;
            width: 100%;
            border-radius: 12px;
            overflow: hidden;
            background: #000;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 480px;
        }

        .video-wrapper img {
            width: 100%;
            height: auto;
            display: block;
            object-fit: contain;
        }

        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 16px;
        }

        .metric-card {
            background: var(--bg-card);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-card);
            border-radius: 14px;
            padding: 18px;
            transition: transform 0.2s, border-color 0.2s;
        }

        .metric-card:hover {
            transform: translateY(-2px);
            border-color: rgba(0, 242, 254, 0.3);
        }

        .metric-label {
            color: var(--text-muted);
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            font-weight: 600;
            margin-bottom: 8px;
        }

        .metric-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--text-primary);
        }

        .metric-sub {
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 4px;
        }

        .pill-active {
            color: var(--accent-green) !important;
        }
        .pill-idle {
            color: var(--text-muted) !important;
        }
        .pill-warn {
            color: var(--accent-orange) !important;
        }

        .progress-bar-bg {
            width: 100%;
            height: 6px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 3px;
            margin-top: 10px;
            overflow: hidden;
        }

        .progress-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--accent-cyan), var(--accent-blue));
            width: 0%;
            transition: width 0.3s ease;
        }
    </style>
</head>
<body>
    <header>
        <div class="header-title">
            <h1>STAGE 1 -- VIDEO INGESTION & TELEMETRY HUD</h1>
            <p>ISRO Space Experiment Monitoring | Human Activity Recognition Pipeline</p>
        </div>
        <div class="status-badge">
            <span class="status-dot"></span>
            LIVE STREAMING
        </div>
    </header>

    <div class="grid-container">
        <!-- Live Video Stream -->
        <div class="video-card">
            <div class="video-wrapper">
                <img id="stream-view" src="/video_feed" alt="Live Stream">
            </div>
        </div>

        <!-- Telemetry Gauges -->
        <div class="sidebar">
            <div class="metric-card">
                <div class="metric-label">Ingestion Rate</div>
                <div class="metric-value" id="val-fps">-- <span style="font-size: 1rem; color: var(--text-muted)">FPS</span></div>
                <div class="metric-sub" id="val-latency">Latency: -- ms</div>
            </div>

            <div class="metric-card">
                <div class="metric-label">Motion Activity Status</div>
                <div class="metric-value" id="val-motion">--</div>
                <div class="progress-bar-bg">
                    <div class="progress-bar-fill" id="bar-motion"></div>
                </div>
                <div class="metric-sub" id="val-motion-score">Energy: 0.0%</div>
            </div>

            <div class="metric-card">
                <div class="metric-label">Frame Sharpness / Quality</div>
                <div class="metric-value" id="val-blur">--</div>
                <div class="metric-sub" id="val-quality">Status: NORMAL</div>
            </div>

            <div class="metric-card">
                <div class="metric-label">Stream Telemetry</div>
                <div class="metric-sub" id="val-frame-count">Frames Processed: 0</div>
                <div class="metric-sub" id="val-resolution">Resolution: --</div>
                <div class="metric-sub" id="val-source">Source: --</div>
            </div>
        </div>
    </div>

    <script>
        function updateTelemetry() {
            fetch('/telemetry')
                .then(r => r.json())
                .then(data => {
                    if (!data || Object.keys(data).length === 0) return;

                    document.getElementById('val-fps').innerHTML = `${data.fps} <span style="font-size: 1rem; color: var(--text-muted)">FPS</span>`;
                    document.getElementById('val-latency').innerText = `Ingestion Latency: ${data.latency_ms} ms`;

                    const motionEl = document.getElementById('val-motion');
                    if (data.motion_detected) {
                        motionEl.innerText = 'ACTIVE';
                        motionEl.className = 'metric-value pill-active';
                    } else {
                        motionEl.innerText = 'IDLE';
                        motionEl.className = 'metric-value pill-idle';
                    }
                    document.getElementById('bar-motion').style.width = `${Math.min(100, data.motion_score * 5)}%`;
                    document.getElementById('val-motion-score').innerText = `Motion Energy: ${data.motion_score}%`;

                    const blurEl = document.getElementById('val-blur');
                    blurEl.innerText = data.blur_score;
                    if (data.is_blurry) {
                        document.getElementById('val-quality').innerText = 'Status: LOW (BLUR DETECTED)';
                        document.getElementById('val-quality').className = 'metric-sub pill-warn';
                    } else {
                        document.getElementById('val-quality').innerText = 'Status: CRISP & HIGH QUALITY';
                        document.getElementById('val-quality').className = 'metric-sub pill-active';
                    }

                    document.getElementById('val-frame-count').innerText = `Frames Processed: #${data.frame_id}`;
                    document.getElementById('val-resolution').innerText = `Resolution: ${data.resolution[0]}x${data.resolution[1]}`;
                    document.getElementById('val-source').innerText = `Source Type: ${data.source_type}`;
                })
                .catch(e => console.error(e));
        }

        setInterval(updateTelemetry, 250);
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_DASHBOARD)


@app.route("/telemetry")
def telemetry():
    global latest_telemetry
    return jsonify(latest_telemetry)


def generate_mjpeg():
    """Generator yielding MJPEG frame stream."""
    global vim_instance, latest_telemetry
    if vim_instance is None:
        return

    for frame, meta in vim_instance.get_frames(with_metadata=True):
        latest_telemetry = meta.to_dict()

        # Convert RGB to BGR for JPEG compression
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ret, jpeg = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")


@app.route("/video_feed")
def video_feed():
    return Response(generate_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")


def main():
    global vim_instance
    parser = argparse.ArgumentParser(description="Stage 1 Telemetry Web HUD")
    parser.add_argument("--source", type=str, default="0", help="Webcam (0), video file path, or RTSP/HTTP URL")
    parser.add_argument("--port", type=int, default=5000, help="Web server port")
    parser.add_argument("--normalize", action="store_true", help="Enable CLAHE normalization")
    parser.add_argument("--zero-latency", action="store_true", help="Enable zero-latency mode")
    parser.add_argument("--motion-adaptive", action="store_true", help="Enable motion-adaptive sampling")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    vim_instance = VideoInputModule(
        source=source,
        width=640,
        height=480,
        normalize=args.normalize,
        zero_latency=args.zero_latency,
        motion_adaptive=args.motion_adaptive,
    )

    if not vim_instance.start():
        print("[ERROR] Failed to start video capture.")
        return

    print(f"\n[INFO] Mission Telemetry Web HUD active!")
    print(f"[INFO] Open in your browser: http://localhost:{args.port}\n")

    try:
        app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
    finally:
        vim_instance.stop()


if __name__ == "__main__":
    main()
