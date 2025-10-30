import os, io, time, subprocess, signal, threading, json
from contextlib import contextmanager

import numpy as np
import cv2 as cv
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse, JSONResponse
from ultralytics import YOLO
import requests
from collections import deque  

# ------------------------------------------------------------
# Config 
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DEFAULT_MODEL_ACC = os.path.join(PROJECT_ROOT, "models", "accident-yolov11n.pt")
DEFAULT_MODEL_POT = os.path.join(PROJECT_ROOT, "models", "pothole-yolo.pt")

MODEL_PATH_ACC = os.getenv("MODEL_PATH", DEFAULT_MODEL_ACC)
MODEL_PATH_POT = os.getenv("POTHOLE_MODEL_PATH", DEFAULT_MODEL_POT)

IMG_SIZE  = int(os.getenv("IMG_SIZE", "640"))
CONF_ACC  = float(os.getenv("CONF_ACCIDENT", "0.75"))
CONF_POT  = float(os.getenv("CONF_POTHOLE",  "0.75"))
SKIP_N    = int(os.getenv("DETECT_EVERY_N", "1"))   

# --- filtering params  ---
VOTE_WIN      = int(os.getenv("VOTE_WIN", "8"))
VOTE_K        = int(os.getenv("VOTE_K", "4"))
COOLDOWN_S    = int(os.getenv("COOLDOWN_SEC", "600"))
MIN_IOU_SAME  = float(os.getenv("MIN_IOU_SAME_EVENT", "0.5"))

# --- backend posting target ---
# BACKEND_URL=http://127.0.0.1:8000 / http://192.168.1.50:8000
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
CAMERA_ID   = os.getenv("CAMERA_ID", "pi_cam01")

# ------------------------------------------------------------
# Simple drawing helper
def draw_boxes(frame, dets, label):
    color = (0, 255, 0) if label == "Accident" else (255, 0, 0)
    for (x1, y1, x2, y2, conf, _name, _cls) in dets:
        cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv.putText(frame, f"{label} {conf:.2f}", (x1, max(10, y1 - 6)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv.LINE_AA)
    return frame

def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    inter = max(0, min(ax2,bx2)-max(ax1,bx1)) * max(0, min(ay2,by2)-max(ay1,by1))
    ia = max(0,(ax2-ax1)) * max(0,(ay2-ay1))
    ib = max(0,(bx2-bx1)) * max(0,(by2-by1))
    return inter / (ia + ib - inter + 1e-6)

def union_box(boxes_xyxy):
    x1 = min(b[0] for b in boxes_xyxy)
    y1 = min(b[1] for b in boxes_xyxy)
    x2 = max(b[2] for b in boxes_xyxy)
    y2 = max(b[3] for b in boxes_xyxy)
    return (x1,y1,x2,y2)

def _encode_jpeg(frame, q=80):
    ok, buf = cv.imencode(".jpg", frame, [int(cv.IMWRITE_JPEG_QUALITY), q])
    return buf.tobytes() if ok else None

def _post_event(kind: str, camera_id: str, bbox_xyxy, mean_conf, frame, lat=None, lon=None):
    """
    POST confirmed event to backend:
      /ingest/accident or /ingest/pothole
    """
    url = f"{BACKEND_URL}/ingest/{kind}"
    jpg = _encode_jpeg(frame, q=85)
    files = {"image": ("frame.jpg", jpg, "image/jpeg")}
    data = {
        "camera_id": camera_id,
        "mean_conf": str(float(mean_conf)),
        "bbox_xyxy": json.dumps([int(v) for v in bbox_xyxy]),
    }
    if lat is not None: data["latitude"]  = str(lat)
    if lon is not None: data["longitude"] = str(lon)
    try:
        r = requests.post(url, data=data, files=files, timeout=10)
        print(f"[POST {kind}] {r.status_code} {r.text[:120]}")
    except Exception as e:
        print(f"[POST {kind}] failed: {e}")

# ------------------------------------------------------------
# Model handler
class PiDetectors:
    def __init__(self, imgsz):
        self.imgsz = imgsz
        self.model_acc = None
        self.model_pot = None
        self.names_acc = None
        self.names_pot = None
        self._lock = threading.Lock()

    def load(self):
        with self._lock:
            if self.model_acc is None:
                print(f"[boot] loading accident model: {MODEL_PATH_ACC}")
                self.model_acc = YOLO(MODEL_PATH_ACC)
            if self.model_pot is None:
                print(f"[boot] loading pothole model: {MODEL_PATH_POT}")
                self.model_pot = YOLO(MODEL_PATH_POT)
        print("[boot] models loaded.")

    def detect(self, img_bgr, conf_acc=CONF_ACC, conf_pot=CONF_POT):
        # Ensure models are ready
        if self.model_acc is None or self.model_pot is None:
            self.load()

        res_acc = self.model_acc.predict(
            img_bgr, imgsz=self.imgsz, conf=conf_acc, verbose=False
        )[0]
        res_pot = self.model_pot.predict(
            img_bgr, imgsz=self.imgsz, conf=conf_pot, verbose=False
        )[0]

        if self.names_acc is None: self.names_acc = res_acc.names
        if self.names_pot is None: self.names_pot = res_pot.names

        acc, pot = [], []
        for b in res_acc.boxes:
            cls_id = int(b.cls)
            conf_f = float(b.conf)
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            name = self.names_acc[cls_id]
            acc.append((x1, y1, x2, y2, conf_f, name, cls_id))

        for b in res_pot.boxes:
            cls_id = int(b.cls)
            conf_f = float(b.conf)
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            name = self.names_pot[cls_id]
            pot.append((x1, y1, x2, y2, conf_f, name, cls_id))

        return acc, pot

det = PiDetectors(IMG_SIZE)

# ------------------------------------------------------------
# MJPEG generator from rpicam-vid
# ------------------------------------------------------------
@contextmanager
def rpicam_process(width, height, fps):
    # -rpicam-vid outputs a continuous MJPEG byte stream to stdout with -o -
    cmd = [
        "rpicam-vid",
        "-t", "0",
        "--width", str(width),
        "--height", str(height),
        "--framerate", str(fps),
        "--codec", "mjpeg",
        "-o", "-",
        "--nopreview"
    ]
    print("[rpicam] starting:", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        yield proc
    finally:
        print("[rpicam] terminating process")
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=2)
            except Exception:
                proc.kill()

def mjpeg_frames_from_rpicam(width, height, fps):
    with rpicam_process(width, height, fps) as proc:
        buf = b""
        stream = proc.stdout
        while True:
            chunk = stream.read(8192)
            if not chunk:
                break
            buf += chunk
            s = buf.find(b"\xff\xd8")
            e = buf.find(b"\xff\xd9")
            if s != -1 and e != -1 and e > s:
                jpg = buf[s:e + 2]
                buf = buf[e + 2:]
                frame = cv.imdecode(np.frombuffer(jpg, np.uint8), cv.IMREAD_COLOR)
                if frame is not None:
                    yield frame

# ------------------------------------------------------------
# FastAPI app
app = FastAPI(title="BUAD Pi Camera Detection")

@app.get("/health")
def health():
    ok = os.system("which rpicam-vid > /dev/null 2>&1") == 0
    return {
        "status": "ok",
        "rpicam_vid_found": ok,
        "models": {
            "accident": os.path.exists(MODEL_PATH_ACC),
            "pothole": os.path.exists(MODEL_PATH_POT),
        }
    }

@app.get("/warmup")
def warmup():
    det.load()
    return {"status": "warmed"}

@app.get("/stream")
def stream(
    width: int = Query(1280, ge=160, le=3840),
    height: int = Query(720, ge=120,  le=2160),
    fps: int = Query(15, ge=1, le=60),
    conf_acc: float = Query(CONF_ACC, ge=0.0, le=1.0),
    conf_pot: float = Query(CONF_POT, ge=0.0, le=1.0),
    detect_every_n: int = Query(SKIP_N, ge=1, le=10),

    # GPS passthrough
    lat: float | None = Query(default=24.853780),
    lon: float | None = Query(default=46.711963),

    # camera id override if needed
    camera_id: str = Query(default=CAMERA_ID),
):
    """
    Streams MJPEG with overlays. Runs detection + filtering.
    Sends confirmed accidents/potholes to backend via POST.
    """
    try:
        def gen():
            det.load()

            pred_hist_acc = deque(maxlen=VOTE_WIN)
            last_event_time_acc = 0.0
            last_event_roi_acc  = None

            pred_hist_pot = deque(maxlen=VOTE_WIN)
            last_event_time_pot = 0.0
            last_event_roi_pot  = None

            last = time.time()
            frame_i = 0

            for frame in mjpeg_frames_from_rpicam(width, height, fps):
                frame_i += 1
                vis = frame

                # detect every N frames
                if frame_i % detect_every_n == 0:
                    try:
                        acc, pot = det.detect(frame, conf_acc=conf_acc, conf_pot=conf_pot)

                        # drawing overlays at the stream
                        if acc: vis = draw_boxes(vis, acc, "Accident")
                        if pot: vis = draw_boxes(vis, pot, "Pothole")

                        #  Accident filtering (vote + cooldown + IoU) 
                        pred_hist_acc.append(1 if acc else 0)
                        if acc and sum(pred_hist_acc) >= VOTE_K:
                            union = union_box([(d[0], d[1], d[2], d[3]) for d in acc])
                            now = time.time()
                            same_time = (now - last_event_time_acc) < COOLDOWN_S
                            same_roi  = (last_event_roi_acc is not None) and iou(last_event_roi_acc, union) >= MIN_IOU_SAME

                            if not (same_time or same_roi):
                                avg_conf = float(sum(d[4] for d in acc)/len(acc))
                                main = max(acc, key=lambda d: d[4])
                                bbox = (main[0], main[1], main[2], main[3])
                                _post_event("accident", camera_id, bbox, avg_conf, frame, lat, lon)
                                last_event_time_acc, last_event_roi_acc = now, union
                                pred_hist_acc.clear()

                        # Pothole filtering (the same as accidents :VOTING + cooldown + IoU) 
                        pred_hist_pot.append(1 if pot else 0)
                        if pot and sum(pred_hist_pot) >= VOTE_K:
                            union_p = union_box([(d[0], d[1], d[2], d[3]) for d in pot])
                            now = time.time()
                            same_time_p = (now - last_event_time_pot) < COOLDOWN_S
                            same_roi_p  = (last_event_roi_pot is not None) and iou(last_event_roi_pot, union_p) >= MIN_IOU_SAME

                            if not (same_time_p or same_roi_p):
                                avg_conf = float(sum(d[4] for d in pot)/len(pot))
                                main = max(pot, key=lambda d: d[4])
                                bbox = (main[0], main[1], main[2], main[3])
                                _post_event("pothole", camera_id, bbox, avg_conf, frame, lat, lon)
                                last_event_time_pot, last_event_roi_pot = now, union_p
                                pred_hist_pot.clear()

                    except Exception as e:
                        print("[detect] error:", e)

                # FPS overlay for stream
                now = time.time()
                cur_fps = 1.0 / max(1e-6, now - last)
                last = now
                cv.putText(vis, f"FPS {cur_fps:.1f}", (10, 24),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv.LINE_AA)

                ok, jpg = cv.imencode(".jpg", vis, [int(cv.IMWRITE_JPEG_QUALITY), 80])
                if not ok:
                    continue
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
                       jpg.tobytes() + b"\r\n")

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")
    except FileNotFoundError:
        return JSONResponse(status_code=500, content={
            "error": "rpicam-vid not found. Install libcamera-apps and try again."
        })

if __name__ == "__main__":
    import uvicorn
    print("Starting BUAD Pi server on :9000")
    uvicorn.run("test:app", host="0.0.0.0", port=9000, log_level="info")
