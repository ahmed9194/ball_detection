"""
Ball Tracker — YOLOv5n (COCO pretrained, class 32 = sports ball)
-----------------------------------------------------------------
Laptop testing  : python ball_tracker.py
ESP32 export    : python ball_tracker.py --export
"""

import argparse
import os
import time
import threading
import requests
from collections import deque
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import torch

# ── Config ────────────────────────────────────────────────────────────────────
YOLOV5_REPO = "ultralytics/yolov5:v7.0"  # pin to avoid breaking changes on master
BALL_CLASS   = 32       # COCO index for "sports ball"
CONF_THRESH  = 0.30     # detection confidence gate
IOU_THRESH   = 0.45     # NMS overlap threshold
INPUT_SIZE   = 320      # inference resolution — 320 fastest, 640 more accurate
SMOOTH_LEN   = 5        # frames to smooth position over
TRAIL_LEN    = 30       # frames of motion trail
DEFAULT_STREAM_URL = "http://10.54.61.164/" 
DEFAULT_SENSOR_URL = "http://10.54.61.164:82/distance"

# ── Colours (BGR) ─────────────────────────────────────────────────────────────
COL_BOX    = (0, 220, 50)
COL_CENTER = (0, 0, 255)
COL_TRAIL  = (255, 180, 0)
COL_LOST   = (60, 60, 200)


class BallTracker:
    def __init__(self):
        print("[INFO] Loading YOLOv5n from PyTorch Hub …")
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        try:
            self.model = torch.hub.load(
                YOLOV5_REPO,
                "yolov5n",
                pretrained=True,
                force_reload=False,
                trust_repo=True,
                verbose=False,
            )
        except ModuleNotFoundError as exc:
            if exc.name == "ultralytics":
                raise RuntimeError(
                    "Missing dependency 'ultralytics' while loading YOLOv5 from torch.hub. "
                    "Run: pip install ultralytics"
                ) from exc
            raise
        self.model.eval()
        self.model.classes = [BALL_CLASS]
        self.model.conf    = CONF_THRESH
        self.model.iou     = IOU_THRESH
        self.model.max_det = 5

        # Warm-up (avoids slow first frame)
        dummy = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
        self.model(dummy, size=INPUT_SIZE)
        print("[INFO] Model ready.\n")

        self.history:    deque = deque(maxlen=SMOOTH_LEN)
        self.trail:      deque = deque(maxlen=TRAIL_LEN)
        self.lost_frames: int  = 0
        self.fps_buf:    deque = deque(maxlen=20)
        self._conf             = CONF_THRESH
        
        # --- NEW: Sensor Distance Variables & Thread ---
        self.current_distance = "Fetching..."
        self.sensor_thread = threading.Thread(target=self._update_distance_loop, daemon=True)
        self.sensor_thread.start()

    # --- NEW: Background Worker for Distance ---
    def _update_distance_loop(self):
        """Constantly fetches distance from the ESP32 in the background without lagging the video."""
        while True:
            try:
                resp = requests.get(DEFAULT_SENSOR_URL, timeout=0.5)
                if resp.status_code == 200:
                    self.current_distance = resp.text.strip()
            except requests.exceptions.RequestException:
                self.current_distance = "Network Error"
            time.sleep(0.1)

    # ── Detection ─────────────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray):
        self.model.conf = self._conf
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(rgb, size=INPUT_SIZE)

        df = results.pandas().xyxy[0]
        df = df[df["class"] == BALL_CLASS]
        if df.empty:
            return None

        best = df.loc[df["confidence"].idxmax()]
        x1, y1, x2, y2 = int(best.xmin), int(best.ymin), int(best.xmax), int(best.ymax)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        bw = x2 - x1
        bh = y2 - y1
        return cx, cy, bw, bh, float(best.confidence)

    # ── Smoothing ─────────────────────────────────────────────────────────────
    def smooth(self, det):
        self.history.append(det)
        pts = list(self.history)
        w   = np.exp(np.linspace(0, 1, len(pts)))
        w  /= w.sum()
        cx  = int(sum(wi * p[0] for wi, p in zip(w, pts)))
        cy  = int(sum(wi * p[1] for wi, p in zip(w, pts)))
        bw  = int(sum(wi * p[2] for wi, p in zip(w, pts)))
        bh  = int(sum(wi * p[3] for wi, p in zip(w, pts)))
        return cx, cy, bw, bh, pts[-1][4]

    # ── Drawing ───────────────────────────────────────────────────────────────
    def _draw_found(self, frame, cx, cy, bw, bh, conf, fps):
        self.trail.append((cx, cy))
        for i in range(1, len(self.trail)):
            t = i / len(self.trail)
            cv2.line(frame, self.trail[i-1], self.trail[i], COL_TRAIL, max(1, int(3 * t)))

        x1, y1 = cx - bw // 2, cy - bh // 2
        x2, y2 = cx + bw // 2, cy + bh // 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), COL_BOX, 2)

        r = max(bw, bh) // 2
        cv2.circle(frame, (cx, cy), 4, COL_CENTER, -1)
        cv2.line(frame, (cx - r, cy), (cx + r, cy), COL_BOX, 1)
        cv2.line(frame, (cx, cy - r), (cx, cy + r), COL_BOX, 1)

        label = f"ball {conf:.2f}  ({cx},{cy})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), COL_BOX, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        self._draw_hud(frame, fps)

    def _draw_lost(self, frame, fps):
        msg = "Searching..." if self.lost_frames < 10 else "No ball detected"
        cv2.putText(frame, msg, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COL_LOST, 2)
        self._draw_hud(frame, fps)

    def _draw_hud(self, frame, fps):
        # Existing Top Data
        cv2.putText(
            frame,
            f"FPS:{fps:.1f}  conf:{self._conf:.2f}  [+/-]  YOLOv5n",
            (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2,
        )
        
        # --- NEW: Overlay the Real-Time Distance Data ---
        cv2.putText(
            frame,
            f"Target Distance: {self.current_distance} cm",
            (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
        )

    # ── Main loop ─────────────────────────────────────────────────────────────
    # (The stream handling methods remain exactly the same as your original code)
    def _candidate_stream_urls(self, source: str):
        p = urlparse(source)
        host = p.hostname
        scheme = p.scheme or "http"
        netloc = p.netloc or ""

        candidates = [source]
        base_urls = []
        if netloc:
            base_urls.append(f"{scheme}://{netloc}")
        if host:
            base_urls.append(f"{scheme}://{host}")

        # Keep source first, then expand likely stream endpoints from both host and netloc.
        for base in base_urls:
            candidates.extend([
                f"{base}/stream",
                f"{base}/video",
                f"{base}/mjpeg",
            ])

        if host:
            candidates.extend([
                f"{scheme}://{host}:81/stream",
                f"{scheme}://{host}:81/video",
            ])

        deduped = []
        seen = set()
        for url in candidates:
            if url not in seen:
                seen.add(url)
                deduped.append(url)
        return deduped

    def _candidate_snapshot_urls(self, source: str):
        p = urlparse(source)
        host = p.hostname
        scheme = p.scheme or "http"
        netloc = p.netloc or ""

        candidates = [source]
        base_urls = []
        if netloc:
            base_urls.append(f"{scheme}://{netloc}")
        if host:
            base_urls.append(f"{scheme}://{host}")

        # Even when source is /stream, still probe JPEG snapshot endpoints on the same host.
        for base in base_urls:
            candidates.extend([
                f"{base}/capture",
                f"{base}/jpg",
                f"{base}/snapshot",
            ])

        deduped = []
        seen = set()
        for url in candidates:
            if url not in seen:
                seen.add(url)
                deduped.append(url)
        return deduped

    def _probe_snapshot_url(self, source: str):
        tried = []
        for url in self._candidate_snapshot_urls(source):
            tried.append(url)
            try:
                resp = requests.get(url, timeout=2.0)
                if resp.status_code != 200 or not resp.content:
                    continue
            except requests.exceptions.RequestException:
                continue

            frame = cv2.imdecode(np.frombuffer(resp.content, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None and frame.size > 0:
                return url, tried
        return None, tried

    def _read_snapshot_frame(self, snapshot_url: str):
        try:
            resp = requests.get(snapshot_url, timeout=2.0)
            if resp.status_code != 200 or not resp.content:
                return None
            data = resp.content
        except requests.exceptions.RequestException:
            return None
        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        return frame

    def _open_source(self, source):
        if isinstance(source, str) and source.startswith(("http://", "https://")):
            tried = []
            for url in self._candidate_stream_urls(source):
                for backend in (None, cv2.CAP_FFMPEG):
                    cap = cv2.VideoCapture(url) if backend is None else cv2.VideoCapture(url, backend)
                    if cap.isOpened():
                        print(f"[INFO] Using stream source: {url}")
                        return cap, url
                    tried.append(url if backend is None else f"{url} [FFMPEG]")
                    cap.release()
            raise RuntimeError("Cannot open camera stream. Tried:\n  - " + "\n  - ".join(tried))

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {source}")
        return cap, source

    def run(self, source=DEFAULT_STREAM_URL):
        cap = None
        snapshot_url = None
        if isinstance(source, str) and source.startswith(("http://", "https://")):
            snapshot_url, snapshot_tried = self._probe_snapshot_url(source)
            if snapshot_url is not None:
                print(f"[INFO] Snapshot mode active: {snapshot_url}")
            else:
                try:
                    cap, used = self._open_source(source)
                    print(f"[INFO] OpenCV stream mode active: {used}")
                except RuntimeError as stream_error:
                    raise RuntimeError(
                        f"{stream_error}\nAlso tried snapshot endpoints:\n  - "
                        + "\n  - ".join(snapshot_tried)
                    ) from stream_error
        else:
            cap, _ = self._open_source(source)

        print("Controls:  'q' quit   '+' raise conf   '-' lower conf\n")

        while True:
            t0 = time.perf_counter()
            if snapshot_url is not None:
                frame = self._read_snapshot_frame(snapshot_url)
                if frame is None:
                    self.lost_frames += 1
                    fps = self._tick(t0)
                    black = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
                    self._draw_lost(black, fps)
                    cv2.putText(black, "Snapshot fetch failed", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_LOST, 2)
                    cv2.imshow("Ball Tracker — YOLOv5n", black)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"): break
                    continue
            else:
                ret, frame = cap.read()
                if not ret: break

            det = self.detect(frame)
            fps = self._tick(t0)

            if det is not None:
                self.lost_frames = 0
                cx, cy, bw, bh, conf = self.smooth(det)
                self._draw_found(frame, cx, cy, bw, bh, conf, fps)
            else:
                self.lost_frames += 1
                self.history.clear()
                self._draw_lost(frame, fps)

            cv2.imshow("Ball Tracker — YOLOv5n", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("+"):
                self._conf = min(0.95, round(self._conf + 0.05, 2))
            elif key == ord("-"):
                self._conf = max(0.05, round(self._conf - 0.05, 2))

        if cap is not None: cap.release()
        cv2.destroyAllWindows()

    def _tick(self, t0):
        self.fps_buf.append(1.0 / max(time.perf_counter() - t0, 1e-9))
        return float(np.mean(self.fps_buf))

    # ── ESP32 export ──────────────────────────────────────────────────────────
    def export_tflite(self):
        print("[EXPORT] Step 1 — exporting to ONNX …")
        import subprocess, sys
        onnx_path = Path("yolov5n_ball.onnx")
        subprocess.run([
            sys.executable, "-c",
            f"""
import torch
model = torch.hub.load('{YOLOV5_REPO}', 'yolov5n', pretrained=True, trust_repo=True, verbose=False)
model.eval()
dummy = torch.zeros(1, 3, {INPUT_SIZE}, {INPUT_SIZE})
torch.onnx.export(
    model, dummy, '{onnx_path}',
    opset_version=12,
    input_names=['images'],
    output_names=['output'],
    dynamic_axes={{'images': {{0: 'batch'}}, 'output': {{0: 'batch'}}}},
)
"""
        ], check=True)

        print("[EXPORT] Step 2 — converting ONNX → TFLite INT8 …")
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf

        tf_path = Path("yolov5n_ball_tf")
        onnx_model = onnx.load(str(onnx_path))
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(str(tf_path))

        converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_path))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type  = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()

        tflite_path = Path("ball_tracker_esp32.tflite")
        tflite_path.write_bytes(tflite_model)
        size_kb = tflite_path.stat().st_size // 1024
        print(f"[EXPORT] Done! → {tflite_path}  ({size_kb} KB)")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv5n Ball Tracker")
    parser.add_argument("--source", default=DEFAULT_STREAM_URL, help="Camera index or stream URL")
    parser.add_argument("--export", action="store_true", help="Export INT8 TFLite model")
    args = parser.parse_args()

    tracker = BallTracker()
    if args.export:
        tracker.export_tflite()
    else:
        source = int(args.source) if str(args.source).isdigit() else args.source
        tracker.run(source=source)