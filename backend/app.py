# Imports

import cv2, os, time, json, uuid, traceback, asyncio
import numpy as np
from datetime import datetime, timezone
from typing import Optional, List
from collections import deque
from threading import Lock, Thread
# from pathlib import Path

#------------------------------
from fastapi import FastAPI, UploadFile, File, Form, Query, Path
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

#------------------------------
import sqlalchemy
from sqlalchemy import text
from supabase import create_client
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.pgvector import PgVector, SearchType
from agno.knowledge.embedder.openai import OpenAIEmbedder

#----------------------------------
# Accident Detector
from backend.accident_detector import AccidentDetector

# VLM
# from backend.vlm import vlm_analyze_image, VLM_agent
from backend.vlm_agent_wrapper import analyze_image_with_vlm  

# DB Insert
from backend.db_insertion import db_insert_accident, db_insert_pothole
#----------------------------------
from dotenv import load_dotenv

# -----------------------------------------------------------------------------------
# Load variables from .env file 
load_dotenv()

# paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(PROJECT_ROOT, "models", "accident-yolov11.pt")
DEFAULT_OUT   = os.path.join(PROJECT_ROOT, "outputs")

# env vars
MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL)
OUT_DIR    = os.getenv("OUT_DIR", DEFAULT_OUT)
IMG_SIZE   = int(os.getenv("IMG_SIZE", "640"))   # model input size
CONF_ACC   = float(os.getenv("CONF_ACCIDENT", "0.75"))  # detection confidence threshold
CONF_POT   = float(os.getenv("CONF_POTHOLE",  "0.75"))  # pothole detection conf

# >> event logic (voting / cooldown / de-dup)
VOTE_WIN   = int(os.getenv("VOTE_WIN", "12"))  # frames window to vote over
VOTE_K     = int(os.getenv("VOTE_K", "6"))     # how many “accident frames” required
COOLDOWN_S = int(os.getenv("COOLDOWN_SEC", "600"))  # prevent duplicate events (seconds)
MIN_IOU_SAME = float(os.getenv("MIN_IOU_SAME_EVENT", "0.5"))  # overlap threshold for same event

os.makedirs(OUT_DIR, exist_ok=True)

# >> DATABASE
DATABASE_URL = os.getenv("DATABASE_URL")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "Capstone")                            

# >> SQLAlchemy engine
engine = sqlalchemy.create_engine(DATABASE_URL)
# >> Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# >> Vector DB
embedder = OpenAIEmbedder()

# Accident Vector DB
try:
    knowledge = Knowledge(
        vector_db=PgVector(
            table_name="Accident_reports",
            db_url=DATABASE_URL,
            embedder=embedder,
            search_type=SearchType.hybrid
        ),
    )
except Exception as e:
    knowledge = None
    print("Warning: Accident Knowledge (vector DB) initialization failed:", e)

# Pothole Vector DB
try:
    knowledge_Pothole = Knowledge(
        vector_db=PgVector(
            table_name="Pothole_report",
            db_url=DATABASE_URL,
            embedder=embedder,
            search_type=SearchType.hybrid
        ),
    )
except Exception as e:
    knowledge_Pothole = None
    print("Warning: Pothole Knowledge (vector DB) initialization failed:", e)

# -----------------------------------------------------------------------------------
# Schemas

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

def draw_boxes(frame, boxes):
    for (x1,y1,x2,y2,conf,name) in boxes:
        cv2.rectangle(frame, (x1,y1),(x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"{name} {conf:.2f}", (x1, max(10,y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
    return frame

def save_thumbnail(path, frame, roi=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = frame.copy()
    if roi:
        x1,y1,x2,y2 = map(int, roi)
        cv2.rectangle(img, (x1,y1),(x2,y2), (0,0,255), 2)
    cv2.imwrite(path, img)
    return path

class EventState:
    def __init__(self, fps, vote_win, out_dir):
        self.fps = fps or 25
        self.pred_hist_acc = deque(maxlen=vote_win)
        self.last_event_time = 0.0
        self.last_event_roi = None
        self.out_dir = out_dir

        self.last_event_time_pot = 0.0
        self.last_event_roi_pot = None

def _encode_jpeg(frame):
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return buf.tobytes() if ok else None

def create_event_id(prefix: str, camera_id: str) -> str:
    now = time.time()
    return f"{prefix}_{camera_id}_{int(now)}"

#--------------------------------------
# >> Supabase Upload Function 
def upload_image_to_supabase(file_or_bytes, dest_path: str) -> str:  

    bucket = supabase.storage.from_(SUPABASE_BUCKET)
    try:
        res = bucket.upload(dest_path, file_or_bytes) 
        print("Upload response:", res)
    except Exception as e:
        raise RuntimeError(f"Supabase upload failed: {e}")

    try:
        signed = bucket.create_signed_url(dest_path, 3600)
        public_url = signed.get("signedURL") or signed.get("signedUrl")
    except Exception as e:
        raise RuntimeError(f"Failed to create signed URL: {e}")
    return public_url

# >> Detection Functions
def _submit_bg(fn, *args, **kwargs):
    t = Thread(target=fn, args=args, kwargs=kwargs, daemon=True)
    t.start()

# >> simple pothole size classifier from bbox area
def classify_pothole_size_from_bbox(bbox_xyxy, frame_shape, small_thr=0.01, large_thr=0.05):
    if not bbox_xyxy or frame_shape is None:
        return None
    x1, y1, x2, y2 = map(int, bbox_xyxy)
    w = max(0, x2 - x1) # width, and max 0 so no minus num
    h = max(0, y2 - y1) # height
    bbox_area = w * h # area = width * height
    H, W = frame_shape[:2]
    frame_area = max(1, W * H)
    ratio = bbox_area / frame_area
    if ratio < small_thr:
        return "small"
    if ratio > large_thr:
        return "large"
    return "medium"

# >> CONFIRMED EVENTS PROCESSORS 

def process_confirmed_accident(frame, acc_boxes, camera_id, latitude=None, longitude=None):
    # >> save thumb, upload, call VLM, insert DB (accident)
    try:
        event_id = create_event_id("accident", camera_id)

        # pick first bbox
        x1,y1,x2,y2,score,_,_ = acc_boxes[0]
        bbox_xyxy = (int(x1),int(y1),int(x2),int(y2))

        # save local thumbnail
        thumb_path = os.path.join(OUT_DIR, f"{event_id}.jpg")
        save_thumbnail(thumb_path, frame, bbox_xyxy)

        # read bytes 
        with open(thumb_path, "rb") as f:
            img_bytes = f.read() 

        # VLM --> returns dict 
        vlm_result = analyze_image_with_vlm(image_bytes=img_bytes, image_id=event_id)  

        # get signed URL
        dest_path = f"{camera_id}/{event_id}.jpg"
        public_url = upload_image_to_supabase(img_bytes, dest_path)  

        # DB insert
        bbox_list = [bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2], bbox_xyxy[3]]
        time_utc = datetime.now(timezone.utc).isoformat()
        image_uuid = str(uuid.uuid4())

        asyncio.run(db_insert_accident(
            engine, knowledge,
            event_id=event_id,
            camera_id=camera_id,
            time_utc=time_utc,
            public_url=public_url,
            image_uuid=image_uuid,
            bbox_xyxy=bbox_list,
            latitude=latitude,
            longitude=longitude,
            vlm_result=vlm_result
        ))

        print(f"accident: stored event_id={event_id}")

    except Exception as e:
        print("Error in _process_confirmed_accident:", e)
        traceback.print_exc()

def process_confirmed_pothole(frame, pot_boxes, camera_id, latitude=None, longitude=None):
    try:
        event_id = create_event_id("pothole", camera_id)

        x1,y1,x2,y2,score,_,_ = pot_boxes[0]
        bbox_xyxy = (int(x1),int(y1),int(x2),int(y2))

        size = classify_pothole_size_from_bbox(bbox_xyxy, frame.shape)

        # save local thumbnail
        thumb_path = os.path.join(OUT_DIR, f"{event_id}.jpg")
        save_thumbnail(thumb_path, frame, bbox_xyxy)

        with open(thumb_path, "rb") as f:
            img_bytes = f.read()  

        dest_path = f"Pothole/{camera_id}/{event_id}.jpg"
        public_url = upload_image_to_supabase(img_bytes, dest_path)  

        time_utc = datetime.now(timezone.utc).isoformat()
        image_uuid = str(uuid.uuid4())
        bbox_list = [bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2], bbox_xyxy[3]]

        asyncio.run(db_insert_pothole(
            engine, knowledge_Pothole,
            event_id=event_id,
            camera_id=camera_id,
            time_utc=time_utc,
            public_url=public_url,
            image_uuid=image_uuid,
            bbox_xyxy=bbox_list,
            latitude=latitude,
            longitude=longitude,
            size=size,
            notes=None
        ))

        print(f"pothole: stored event_id={event_id}, size={size}")

    except Exception as e:
        print("Error in _process_confirmed_pothole:", e)
        traceback.print_exc()

# -----------------------------------------------------------------------------------
# App

app = FastAPI(title="Accident Model Backend", version="0.1.0")

# >> for Streamlit & FastAPI connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],  
)

# >> Serve outputs directory under /files
app.mount("/files", StaticFiles(directory=OUT_DIR), name="files")

# >> load models
detector = AccidentDetector(MODEL_PATH, IMG_SIZE)
pothole_model = None

# >> state 
CAMERA_STATES: dict[str, EventState] = {}    
# >> Stream control 
STREAM_FLAGS: dict[str, bool] = {}
STREAM_LOCK = Lock()

def get_state(camera_id: str, fps: float):
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

# >> Stream control 

def _stream_set(camera_id: str, running: bool):
    with STREAM_LOCK:
        STREAM_FLAGS[camera_id] = running

def _stream_is_running(camera_id: str) -> bool:
    with STREAM_LOCK:
        return STREAM_FLAGS.get(camera_id, False)

# POST http://localhost:8000/stream/start?camera_id=cam01
@app.post("/stream/start")
def stream_start(camera_id: str = Query("cam01")):
    _stream_set(camera_id, True)
    return {"ok": True, "camera_id": camera_id, "running": True}

@app.post("/stream/stop")
# POST http://localhost:8000/stream/stop?camera_id=cam01
def stream_stop(camera_id: str = Query("cam01")):
    _stream_set(camera_id, False)
    return {"ok": True, "camera_id": camera_id, "running": False}

@app.get("/stream/status")
#GET http://localhost:8000/stream/status?camera_id=cam01
def stream_status(camera_id: str = Query("cam01")):
    return {"camera_id": camera_id, "running": _stream_is_running(camera_id)}

# ------------------------------------------------------------
# >> stream generator (reads frames, runs model, draws boxes)
def _stream_generator(src: str, conf_acc: float, conf_pot: float, max_fps: int, camera_id: str):
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

            now = time.time()
            if now - last_t < min_dt:
                continue  # skip frames to match max_fps
            last_t = now

            #------------------
            # >> Accident Detection
            dets = detector.predict_image(frame, conf=conf_acc)
            acc = [d for d in dets if d[5].lower() == "accident"]

            state.pred_hist_acc.append(1 if acc else 0)

            vis = draw_boxes(frame.copy(), [(x1,y1,x2,y2,score,"Accident") for (x1,y1,x2,y2,score,_,_) in acc])

            if acc and sum(state.pred_hist_acc) >= VOTE_K:
                union = union_box([(d[0], d[1], d[2], d[3]) for d in acc])
                same_time = (now - state.last_event_time) < COOLDOWN_S
                same_roi  = (state.last_event_roi is not None) and iou(state.last_event_roi, union) >= MIN_IOU_SAME

                if not (same_time and same_roi):
                    _submit_bg(process_confirmed_accident, vis, acc, camera_id)
                    state.last_event_time, state.last_event_roi = now, union
                    state.pred_hist_acc.clear()


            # --------------------------
            # pothole detection 
            pot = []
            if pothole_model is not None:
                dets_pot = pothole_model.predict_image(frame, conf=conf_pot)
                pot = [d for d in dets_pot if d[5].lower() == "pothole"]

                for (x1,y1,x2,y2,score,_,_) in pot:
                    cv2.rectangle(vis, (x1,y1),(x2,y2), (255,0,0), 2)
                    cv2.putText(vis, f"Pothole {score:.2f}", (x1, max(10,y1-6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)

                if pot :
                    union_pot = union_box([(d[0], d[1], d[2], d[3]) for d in pot])
                    same_time_pot = (now - state.last_event_time_pot) < COOLDOWN_S
                    same_roi_pot  = (state.last_event_roi_pot is not None) and iou(state.last_event_roi_pot, union_pot) >= MIN_IOU_SAME
                    if not (same_time_pot and same_roi_pot):
                        _submit_bg(process_confirmed_pothole, vis, pot, camera_id)
                        state.last_event_time_pot, state.last_event_roi_pot = now, union_pot

            jpg = _encode_jpeg(vis)
            if jpg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
    finally:
        cap.release()

# >> Stream endpoint
@app.get("/stream")
def stream(source: str = "0", camera_id: str = "cam01", conf_acc: float = None, conf_pot: float = None, max_fps: int = 10):
    conf_acc_used = conf_acc if conf_acc is not None else CONF_ACC
    conf_pot_used = conf_pot if conf_pot is not None else CONF_POT
    _stream_set(camera_id, True)
    gen = _stream_generator(source, conf_acc_used,conf_pot_used, max_fps, camera_id)
    return StreamingResponse(gen, media_type="multipart/x-mixed-replace; boundary=frame")

# -------------------------------
# >> List info from database

@app.get("/events")
def list_events(limit: int = Query(50, ge=1, le=500)):
    stmt = text("""
        SELECT
            event_id,
            camera_id,
            time_utc,
            mean_conf,
            thumbnail_url
        FROM public.incidents
        ORDER BY time_utc DESC
        LIMIT :limit
    """)
    with engine.connect() as conn:
        rows = conn.execute(stmt, {"limit": limit}).mappings().all()
    return {"events": [dict(r) for r in rows]}

@app.get("/potholes")
def list_potholes(limit: int = Query(50, ge=1, le=500)):
    stmt = text("""
        SELECT
            event_id,
            camera_id,
            time_utc,
            size,
            thumbnail_url,
            bbox_xyxy
        FROM public.potholes
        ORDER BY time_utc DESC
        LIMIT :limit
    """)
    with engine.connect() as conn:
        rows = conn.execute(stmt, {"limit": limit}).mappings().all()
    return {"potholes": [dict(r) for r in rows]}

@app.get("/events/{event_id}/report")
def get_event_report(event_id: str):
    stmt = text("""
        SELECT event_id, raw_report
        FROM public.incidents
        WHERE event_id = :event_id
        LIMIT 1
    """)
    with engine.connect() as conn:
        row = conn.execute(stmt, {"event_id": event_id}).mappings().first()  

    if not row:
        return JSONResponse(status_code=404, content={"detail": "Event not found"})

    raw = row["raw_report"]
    try:
        parsed = raw if isinstance(raw, dict) else json.loads(raw)  
    except Exception:
        parsed = {"raw_report": raw}

    return {"event_id": row["event_id"], "report": parsed}
