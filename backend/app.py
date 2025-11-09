"""
Real-time Vehicle Counting Web Application using YOLOv8 and Flask.

This script launches a Flask web server that performs real-time vehicle detection
and tracking on a video stream. It serves an annotated video feed and provides
API endpoints to retrieve vehicle counts and performance metrics.

Core Functionality:
- Ingests a video file specified by the VIDEO_PATH environment variable.
- Uses a YOLOv8 model for object detection.
- Employs either the SORT or ByteTrack algorithm for object tracking.
- Draws bounding boxes, track IDs, and trails on each frame.
- Counts vehicles as they cross a predefined horizontal line.
- Streams the processed video to a web browser via an MJPEG stream.
- Provides a REST API for accessing counts, metrics, and resetting the state.

Configuration is managed via environment variables. Key variables include:
- VIDEO_PATH: Path to the input video file.
- MODEL_PATH: Path to the YOLOv8 model file (e.g., yolov8s.pt).
- TRACKER_MODE: The tracking algorithm to use, either "SORT" or "BYTE".
- CONF_THRESH: Confidence threshold for object detection.
- LINE_Y: The vertical position of the counting line.
"""
import os
import time
import threading
from collections import deque, defaultdict

import cv2
import numpy as np
import torch
from flask import Flask, Response, jsonify
from flask_cors import CORS
from ultralytics import YOLO

# Assuming sort_tracker.py containing the Sort class is in the same directory.
from sort_tracker import Sort

# --- Configuration from Environment Variables ---
VIDEO_PATH = os.environ.get("VIDEO_PATH", "../videos/video1.mp4")
MODEL_PATH = os.environ.get("MODEL_PATH", "yolov8s.pt")
MODEL_DEVICE = os.environ.get('MODEL_DEVICE', 'auto').lower()
CONF_THRESH = float(os.environ.get("CONF_THRESH", 0.35))
IOU_THRESH = float(os.environ.get("IOU_THRESH", 0.5))
DEFAULT_LINE_Y = int(os.environ.get("LINE_Y", 360))
TRAIL_LEN = int(os.environ.get("TRAIL_LEN", 20))
RESIZE_WIDTH = int(os.environ.get("RESIZE_WIDTH", 960))
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", 80))
RESET_ON_LOOP = os.environ.get("RESET_ON_LOOP", "false").lower() in {"1", "true", "yes"}
TRACKER_MODE = os.environ.get("TRACKER_MODE", "SORT").upper()

# --- Constants ---
VEHICLE_NAMES = {"car", "truck", "bus", "motorcycle"}
TEXT_COLOR = (255, 255, 255)  # White
LINE_COLOR = (0, 255, 255)   # Cyan
BOX_COLOR = (0, 255, 0)      # Green
ID_COLOR = (255, 0, 0)       # Blue

# --- Application Setup ---
app = Flask(__name__)
# Enable Cross-Origin Resource Sharing for API and video feed routes.
CORS(app, resources={r"/api/*": {"origins": "*"}, r"/video_feed": {"origins": "*"}, r"/health": {"origins": "*"}})

# --- Model and Tracker Initialization ---
# Load the YOLOv8 model from the specified path.
model = YOLO(MODEL_PATH)
# Determine the active device for the model (CUDA or CPU).
if MODEL_DEVICE == 'auto':
    MODEL_DEVICE_ACTIVE = 'cuda' if torch.cuda.is_available() else 'cpu'
else:
    MODEL_DEVICE_ACTIVE = MODEL_DEVICE
# Move the model to the selected device, with a fallback to CPU.
try:
    model.to(MODEL_DEVICE_ACTIVE)
except Exception as exc:
    print(f'[warn] Failed to move model to {MODEL_DEVICE_ACTIVE}: {exc}')
    MODEL_DEVICE_ACTIVE = 'cpu'
    try:
        model.to(MODEL_DEVICE_ACTIVE)
    except Exception as exc_cpu:
        print(f'[warn] Failed to move model to cpu: {exc_cpu}')

# Open the video file for processing.
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

# Initialize the SORT tracker if selected.
mot = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# --- Global State Variables ---
# Dictionary to store vehicle counts, protected by a lock for thread safety.
counts = {"up": 0, "down": 0}
counts_lock = threading.Lock()

# Dictionary to store the history of each tracked object.
track_history = defaultdict(lambda: {
    "centers": deque(maxlen=TRAIL_LEN), # Stores recent center points for drawing trails.
    "last_side": None,                  # 'above' or 'below' the line.
    "last_seen": time.time(),           # Timestamp of the last update.
})

# Dictionary for performance metrics, also protected by a lock.
metrics = {
    "start_time": time.time(),
    "frames": 0,
    "recent_ts": deque(maxlen=120),    # Timestamps of recent frames for FPS calculation.
    "fps": 0.0,
    "inference_ms": deque(maxlen=240), # Inference times for averaging.
    "avg_inference_ms": 0.0,
    "model_device": MODEL_DEVICE_ACTIVE,
}
metrics_lock = threading.Lock()


def preprocess(frame):
    """
    Resizes the frame to a standard width if required.

    Args:
        frame (np.ndarray): The input video frame.

    Returns:
        np.ndarray: The resized frame.
    """
    if RESIZE_WIDTH > 0:
        h, w = frame.shape[:2]
        if w != RESIZE_WIDTH:
            scale = RESIZE_WIDTH / float(w)
            frame = cv2.resize(frame, (RESIZE_WIDTH, int(h * scale)))
    return frame


def update_counts_for_crossing(tid, cy, line_y):
    """
    Updates the up/down counts when a tracked object crosses the line.

    Args:
        tid (int): The track ID of the object.
        cy (int): The current y-coordinate of the object's center.
        line_y (int): The y-coordinate of the counting line.
    """
    info = track_history[tid]
    last_side = info["last_side"]
    # Determine the current side of the line.
    side = "below" if cy > line_y else "above"
    # If this is the first time we see this object, just record its side.
    if last_side is None:
        info["last_side"] = side
        return
    # If the object has crossed the line, update the count.
    if side != last_side:
        direction = "down" if side == "below" and last_side == "above" else "up"
        with counts_lock:
            counts[direction] += 1
        # Update the side for the next frame.
        info["last_side"] = side


def draw_overlays(frame, line_y, tracks):
    """
    Draws all visual annotations onto the frame.

    This includes the counting line, bounding boxes for tracked objects,
    track IDs, object trails, and the current vehicle counts.

    Args:
        frame (np.ndarray): The video frame to draw on.
        line_y (int): The y-coordinate of the counting line.
        tracks (np.ndarray): An array of active tracks from the tracker.

    Returns:
        np.ndarray: The frame with overlays drawn on it.
    """
    h, w = frame.shape[:2]
    # Draw the counting line.
    cv2.line(frame, (0, line_y), (w, line_y), LINE_COLOR, 2)
    cv2.putText(frame, f"Line y={line_y}", (10, max(25, line_y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, LINE_COLOR, 2, cv2.LINE_AA)
    # Draw boxes, IDs, and trails for each active track.
    for x1, y1, x2, y2, tid in tracks.astype(int):
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
        cv2.putText(frame, f"ID {tid}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ID_COLOR, 2, cv2.LINE_AA)
        centers = track_history[tid]["centers"]
        # Draw the trail.
        for i in range(1, len(centers)):
            if centers[i - 1] and centers[i]:
                cv2.line(frame, centers[i - 1], centers[i], (200, 200, 200), 2)
    # Draw the total counts on the screen.
    with counts_lock:
        up, down = counts["up"], counts["down"]
    cv2.putText(frame, f"Up: {up}   Down: {down}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, TEXT_COLOR, 2, cv2.LINE_AA)
    return frame


def bump_metrics(inference_ms=None):
    """
    Updates performance metrics such as FPS and average inference time.

    Args:
        inference_ms (float, optional): The inference time for the current frame in milliseconds.
    """
    now = time.time()
    with metrics_lock:
        metrics["frames"] += 1
        metrics["recent_ts"].append(now)
        # Calculate FPS over a sliding window of recent timestamps.
        if len(metrics["recent_ts"]) >= 2:
            dt = metrics["recent_ts"][-1] - metrics["recent_ts"][0]
            if dt > 0:
                metrics["fps"] = (len(metrics["recent_ts"]) - 1) / dt
        # Update average inference time.
        if inference_ms is not None:
            metrics["inference_ms"].append(inference_ms)
            metrics["avg_inference_ms"] = sum(metrics["inference_ms"]) / len(metrics["inference_ms"])


def reset_state(full_reset_tracker=True):
    """
    Resets the application's state, including counts and track history.

    Args:
        full_reset_tracker (bool): If True and using SORT, re-initializes the tracker instance.
    """
    global mot, track_history
    # Reset counts to zero.
    with counts_lock:
        counts["up"] = 0
        counts["down"] = 0
    # Clear the tracking history.
    track_history = defaultdict(lambda: {
        "centers": deque(maxlen=TRAIL_LEN),
        "last_side": None,
        "last_seen": time.time(),
    })
    # Optionally, create a fresh SORT tracker instance.
    if full_reset_tracker and TRACKER_MODE == "SORT":
        mot = Sort(max_age=20, min_hits=3, iou_threshold=0.3)


def frame_generator():
    """
    A generator function that processes video frames and yields them for streaming.

    This is the core processing loop. It reads frames from the video source,
    performs detection and tracking, updates counts, draws overlays, and
    encodes the frame as a JPEG for the MJPEG stream.
    """
    global cap
    while True:
        ret, frame = cap.read()
        # If the video ends, loop back to the beginning.
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if RESET_ON_LOOP:
                reset_state(full_reset_tracker=True)
            continue

        frame = preprocess(frame)
        line_y = DEFAULT_LINE_Y
        tracks_array = np.empty((0, 5), dtype=np.float32)
        inference_start = time.perf_counter()

        # --- Detection and Tracking Logic ---
        if TRACKER_MODE == "SORT":
            # 1. Run detection.
            results = model(frame, conf=CONF_THRESH, iou=IOU_THRESH, verbose=False)[0]
            dets = []
            if results.boxes is not None and len(results.boxes) > 0:
                # 2. Filter detections for vehicle classes.
                for box in results.boxes:
                    name = results.names[int(box.cls)]
                    if name in VEHICLE_NAMES:
                        xyxy = box.xyxy.cpu().numpy().flatten()
                        p = float(box.conf.cpu())
                        dets.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3], p])
            dets_np = np.array(dets, dtype=np.float32) if dets else np.empty((0, 5), dtype=np.float32)
            # 3. Update SORT tracker with the filtered detections.
            tracks_array = mot.update(dets_np)
        else: # ByteTrack
            # Use YOLO's built-in tracking which uses ByteTrack.
            results = model.track(frame, conf=CONF_THRESH, iou=IOU_THRESH, persist=True, verbose=False)[0]
            byte_tracks = []
            if results.boxes is not None and len(results.boxes) > 0 and results.boxes.id is not None:
                 # Filter tracks for vehicle classes.
                for box in results.boxes:
                    name = results.names[int(box.cls)]
                    if name in VEHICLE_NAMES:
                        xyxy = box.xyxy.cpu().numpy().flatten()
                        tid = int(box.id.cpu())
                        byte_tracks.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3], float(tid)])
            tracks_array = np.asarray(byte_tracks, dtype=np.float32) if byte_tracks else np.empty((0, 5), dtype=np.float32)

        inference_ms = (time.perf_counter() - inference_start) * 1000.0
        now = time.time()
        
        # --- Update Track History and Counts ---
        active_ids = set()
        for x1, y1, x2, y2, tid in tracks_array:
            tid = int(tid)
            active_ids.add(tid)
            # Calculate the center of the bounding box.
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            # Update the history for this track.
            track_history[tid]["centers"].append((cx, cy))
            track_history[tid]["last_seen"] = now
            # Check if the track crossed the line.
            update_counts_for_crossing(tid, cy, line_y)

        # --- Clean Up Stale Tracks ---
        # Remove tracks that haven't been seen for a few seconds.
        stale_cutoff = now - 5.0
        for tid, meta in list(track_history.items()):
            if meta["last_seen"] < stale_cutoff and tid not in active_ids:
                del track_history[tid]

        # --- Encode and Yield Frame ---
        frame = draw_overlays(frame, line_y, tracks_array)
        # Encode the frame as JPEG.
        ok, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            continue
        # Update performance metrics.
        bump_metrics(inference_ms)
        # Yield the frame in the format required for MJPEG streaming.
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")

# --- Flask API Routes ---
@app.route("/video_feed")
def video_feed():
    """Flask route to serve the MJPEG video stream."""
    return Response(frame_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/counts")
def api_counts():
    """Flask route to get the current vehicle counts as JSON."""
    with counts_lock:
        return jsonify({"up": int(counts["up"]), "down": int(counts["down"])})


@app.route("/api/reset")
def api_reset():
    """Flask route to reset the application's state."""
    reset_state(full_reset_tracker=True)
    return jsonify({"status": "ok", "message": "Counts and state reset."})


@app.route("/api/metrics")
def api_metrics():
    """Flask route to get performance and configuration metrics as JSON."""
    with metrics_lock:
        data = {
            "fps": round(metrics["fps"], 2),
            "avg_inference_ms": round(metrics["avg_inference_ms"], 2),
            "model_device": metrics["model_device"],
            "frames_processed": int(metrics["frames"]),
            "uptime_sec": int(time.time() - metrics["start_time"]),
            "tracker_mode": TRACKER_MODE,
            "reset_on_loop": RESET_ON_LOOP,
            "resize_width": RESIZE_WIDTH,
            "jpeg_quality": JPEG_QUALITY,
            "conf_thresh": CONF_THRESH,
            "iou_thresh": IOU_THRESH,
            "line_y": DEFAULT_LINE_Y,
        }
    return jsonify(data)


@app.route("/health")
def health():
    """Flask route for a simple health check."""
    return jsonify({"ok": True, "service": "vehicle-counter", "tracker_mode": TRACKER_MODE})


@app.route("/")
def root():
    """Flask route for the root URL."""
    return jsonify({"ok": True, "message": "Use /video_feed and /api/counts"})


# --- Main Execution Block ---
if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5000"))
    # Set some default environment variables if they are not already defined.
    # This can be useful for running the script directly without a .env file.
    os.environ.setdefault("RESIZE_WIDTH", "960")
    os.environ.setdefault("JPEG_QUALITY", "80")
    os.environ.setdefault("CONF_THRESH", "0.35")
    os.environ.setdefault("IOU_THRESH", "0.50")
    # Start the Flask development server.
    app.run(host=host, port=port, debug=False, threaded=True)