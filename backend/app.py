# Imports

import cv2, os, time, json
import numpy as np
from typing import Optional, List
from collections import deque
from threading import Lock

#------------------------------
from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

#----------------------------------
# Accident Detector
from backend.accident_detector import AccidentDetector
# Vision Language Model
from backend.vlm import vlm_analyze_image, VLM_agent 
#----------------------------------


from dotenv import load_dotenv
# Load variables from .env file 
load_dotenv()

# -----------------------------------------------------------------------------------
# FOR DATABSE SETUP 
# db_url = ""   # >> database connection URL
# db_schema = "public"                       # >> schema
# events_table = "events"    ?                # >> metadata & dashboard list 
# vlm_results_table = "vlm_results"          # >> parsed VLM fields 
# mongo_url = ""                             # >> MongoDB URL (for raw images)
# mongo_db = ""                              # >> database name (e.g., "accidents")

# -----------------------------------------------------------------------------------
# Config

# paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(PROJECT_ROOT, "models", "accident-yolov11.pt")
DEFAULT_OUT   = os.path.join(PROJECT_ROOT, "outputs")

# env vars
MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL)
OUT_DIR    = os.getenv("OUT_DIR", DEFAULT_OUT)
IMG_SIZE   = int(os.getenv("IMG_SIZE", "640"))  # model input size
CONF_ACC   = float(os.getenv("CONF_ACCIDENT", "0.40")) # detection confidence threshold

# >> event logic (voting / cooldown / de-dup)
VOTE_WIN   = int(os.getenv("VOTE_WIN", "12"))   # frames window to vote over
VOTE_K     = int(os.getenv("VOTE_K", "6"))      # how many “accident frames” required
COOLDOWN_S = int(os.getenv("COOLDOWN_SEC", "600"))  # prevent duplicate events (seconds)
MIN_IOU_SAME = float(os.getenv("MIN_IOU_SAME_EVENT", "0.5"))  # overlap threshold for same event

os.makedirs(OUT_DIR, exist_ok=True)


# -----------------------------------------------------------------------------------
# Schemas (responses)

class BBox(BaseModel):
    x1: int; y1: int; x2: int; y2: int
    conf: float
    cls: int
    name: str

class ImageResponse(BaseModel):
    detections: List[BBox]
    annotated_path: Optional[str] = None

class ReportResponse(BaseModel):
    event_id: str
    report_text: str
    cached: bool = False

# -----------------------------------------------------------------------------------
# Utils

def iou(a, b):
    # >> box IoU (x1,y1,x2,y2) — used to check “same area”
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    inter = max(0, min(ax2,bx2)-max(ax1,bx1)) * max(0, min(ay2,by2)-max(ay1,by1))
    ia = max(0,(ax2-ax1)) * max(0,(ay2-ay1))
    ib = max(0,(bx2-bx1)) * max(0,(by2-by1))
    return inter / (ia + ib - inter + 1e-6)

def union_box(boxes_xyxy):
    # >> combine all accident boxes into one area
    x1 = min(b[0] for b in boxes_xyxy)
    y1 = min(b[1] for b in boxes_xyxy)
    x2 = max(b[2] for b in boxes_xyxy)
    y2 = max(b[3] for b in boxes_xyxy)
    return (x1,y1,x2,y2)

def draw_boxes(frame, boxes):
    # >> draw boxes on frame
    for (x1,y1,x2,y2,conf,name) in boxes:
        cv2.rectangle(frame, (x1,y1),(x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"{name} {conf:.2f}", (x1, max(10,y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
    return frame

def save_thumbnail(path, frame, roi=None):
    # >> save a small image for dashboard
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = frame.copy()
    if roi:
        x1,y1,x2,y2 = map(int, roi)
        cv2.rectangle(img, (x1,y1),(x2,y2), (0,0,255), 2)
    cv2.imwrite(path, img)
    return path

class EventState:
    # >> store per-camera state (voting, timing, ....)
    def __init__(self, fps, vote_win, out_dir):
        self.fps = fps or 25
        self.pred_hist = deque(maxlen=vote_win)
        self.last_event_time = 0.0
        self.last_event_roi = None
        self.out_dir = out_dir

def _encode_jpeg(frame):
    # >> encode numpy frame to jpeg bytes 
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return buf.tobytes() if ok else None


# -----------------------------------------------------------------------------------
# App

app = FastAPI(title="Accident Model Backend", version="0.1.0")

# >> for Streamlit & FastAPI connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=[""], allow_headers=[""],
)

# >> Serve outputs directory under /files
app.mount("/files", StaticFiles(directory=OUT_DIR), name="files")

# >> load model
detector = AccidentDetector(MODEL_PATH, IMG_SIZE)

# >> state 
CAMERA_STATES: dict[str, EventState] = {}     # per-camera info 

# "TEMPORARY" storage until DB 
# in-memory
EVENTS: dict[str, dict] = {}                  # all accident events 
EVENTS_LOCK = Lock()                        
EVENTS_JSON = os.path.join(OUT_DIR, "events.json")  # events json file

# functions
def _load_events():
    # FOR DB SETUP : for dashboard, fetching recent events
    # >> load the prev events if there is
    if os.path.exists(EVENTS_JSON):
        try:
            data = json.load(open(EVENTS_JSON, "r", encoding="utf-8"))
            if isinstance(data, dict):
                EVENTS.update(data)
        except Exception:
            pass

def _save_events():
    # FOR DATABSE SETUP 
    # >> save events 
    with open(EVENTS_JSON, "w", encoding="utf-8") as f:
        json.dump(EVENTS, f, ensure_ascii=False, indent=2)

_load_events()

def get_state(camera_id: str, fps: float):
    # >> return camera state ,if its not exist creates new one
    st = CAMERA_STATES.get(camera_id)
    if st is None:
        st = EventState(fps=fps or 25, vote_win=VOTE_WIN, out_dir=OUT_DIR)
        CAMERA_STATES[camera_id] = st
    return st


#---------------------------------------
# >> System health check
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH}

# >> Predict image
@app.post("/predict/image", response_model=ImageResponse)
async def predict_image(file: UploadFile = File(...), conf: float = Form(None)):
    data = await file.read() 
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)  # convert to image array

    # run model
    dets = detector.predict_image(img, conf=(conf or CONF_ACC))

    # format bounding boxes
    boxes = [BBox(x1=d[0], y1=d[1], x2=d[2], y2=d[3], conf=d[4], cls=d[6], name=d[5]) for d in dets]

    # draw the boxes and save it
    out_path = os.path.join(OUT_DIR, f"img_{int(time.time())}.jpg")
    vis = draw_boxes(img.copy(), [(b.x1, b.y1, b.x2, b.y2, b.conf, b.name) for b in boxes])
    cv2.imwrite(out_path, vis)

    return ImageResponse(detections=boxes, annotated_path=f"/files/{os.path.basename(out_path)}")

# >> List all accident events (for dashboard)
@app.get("/events")
def list_events(limit: int = Query(50, ge=1, le=500)):
    # FOR DATABSE SETUP 
    with EVENTS_LOCK:
        items = sorted(EVENTS.values(), key=lambda e: e["time_utc"], reverse=True)[:limit] 
        return {"events": items}

# >> VLM Report
@app.get("/events/{event_id}/report", response_model=ReportResponse)
def get_event_report(event_id: str):
    # FOR DATABSE SETUP 
    ev = EVENTS.get(event_id)
    if not ev:
        return JSONResponse(status_code=404, content={"detail": "Event not found"})

    # serve cached report 
    txt = ev.get("report")
    if isinstance(txt, str) and txt.strip():
        return ReportResponse(event_id=event_id, report_text=txt, cached=True)

    # pick the accident thumbnail
    thumb = ev.get("thumbnail")
    if thumb and os.path.exists(thumb):
        txt = vlm_analyze_image(thumb)
    else:
        # no image available 
        return JSONResponse(status_code=400, content={"detail": "No image available"})

    # FOR DATABSE SETUP 
    # store + persist 
    with EVENTS_LOCK:
        EVENTS[event_id]["report"] = txt
        _save_events()

    return ReportResponse(event_id=event_id, report_text=txt, cached=False)


# >> Stream control 
STREAM_FLAGS: dict[str, bool] = {}
STREAM_LOCK = Lock()

def _stream_set(camera_id: str, running: bool):
    with STREAM_LOCK:
        STREAM_FLAGS[camera_id] = running

def _stream_is_running(camera_id: str) -> bool:
    with STREAM_LOCK:
        return STREAM_FLAGS.get(camera_id, False)

@app.post("/stream/start")
def stream_start(camera_id: str = Query(...)):
    _stream_set(camera_id, True)
    return {"ok": True, "camera_id": camera_id, "running": True}

@app.post("/stream/stop")
def stream_stop(camera_id: str = Query(...)):
    _stream_set(camera_id, False)
    return {"ok": True, "camera_id": camera_id, "running": False}

@app.get("/stream/status")
def stream_status(camera_id: str = Query(...)):
    return {"camera_id": camera_id, "running": _stream_is_running(camera_id)}

# >> stream generator (reads frames, runs model, draws boxes)
# reads frames, runs model, draws boxes
def _stream_generator(src: str, conf: float, max_fps: int, camera_id: str):
    # source can be camera '0' or 
    cap = cv2.VideoCapture(int(src)) if src.isdigit() else cv2.VideoCapture(src)
    if not cap.isOpened():
        yield b""  # if cannot open source
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    state = get_state(camera_id, fps)
    min_dt, last_t = 1.0 / max(1, max_fps), 0.0  # limit FPS 

    try:
        while _stream_is_running(camera_id):
            ok, frame = cap.read()
            if not ok:
                break

            # control how many frames per seconds
            now = time.time()
            if now - last_t < min_dt:
                continue  # skip frames to match max_fps
            last_t = now

            dets = detector.predict_image(frame, conf=conf)
            acc = [d for d in dets if d[5].lower() == "accident"]
            # >> update frame prediction history (1 = accident detected, 0 = no accident)
            # # its voting over several frames before confirming accident 
            # # this help us avoids false alerts (Need to be adjusted)
            state.pred_hist.append(1 if acc else 0)

            # draw boxes on the current frame
            vis = draw_boxes(frame.copy(), [(x1,y1,x2,y2,score,"Accident") for (x1,y1,x2,y2,score,_,_) in acc])

            # create new event 
            if acc and sum(state.pred_hist) >= VOTE_K:

                union = union_box([(d[0], d[1], d[2], d[3]) for d in acc])
                same_time = (now - state.last_event_time) < COOLDOWN_S
                same_roi  = (state.last_event_roi is not None) and iou(state.last_event_roi, union) >= MIN_IOU_SAME

                if not (same_time and same_roi):
                    thumb_path = os.path.join(OUT_DIR, f"{camera_id}_{int(now)}.jpg")
                    save_thumbnail(thumb_path, vis, (acc[0][0], acc[0][1], acc[0][2], acc[0][3]))
                    
                    ev = {
                        "event_id": f"{camera_id}_{int(now)}",
                        "type": "accident",
                        "camera_id": camera_id,
                        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
                        "bbox_xyxy": (acc[0][0], acc[0][1], acc[0][2], acc[0][3]),
                        "mean_conf": float(np.mean([d[4] for d in acc])),
                        "thumbnail": thumb_path,
                        "thumbnail_url": f"/files/{os.path.basename(thumb_path)}",
                        "status": "NEW",
                    }

                # # FOR DATABSE SETUP : save/ipdate a new event record (metadata)
                # update state + save event
                #state.last_event_time, state.last_event_roi = now, union
                    with EVENTS_LOCK:
                        EVENTS[ev["event_id"]] = ev
                        _save_events()

            # convert to JPEG and yield chunk
            jpg = _encode_jpeg(vis)
            if jpg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
    finally:
        cap.release()


# >> Stream endpoint
@app.get("/stream")
def stream(source: str = "0", camera_id: str = "cam01", conf: float = None, max_fps: int = 10):
    # >> call example:
    #    /stream?source=0&camera_id=cam01&max_fps=0   (no throttle)
    #    /stream?source=rtsp://<ip>:8554/cam&camera_id=gate1&max_fps=10
    # source might be 0 "Labtop cam" or Rasppery Cam "libcamera0" 
    conf_used = conf if conf is not None else CONF_ACC
    _stream_set(camera_id, True)
    gen = _stream_generator(source, conf_used, max_fps, camera_id)
    return StreamingResponse(gen, media_type="multipart/x-mixed-replace; boundary=frame")


