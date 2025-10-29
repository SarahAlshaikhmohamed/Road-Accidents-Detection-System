# Imports

import cv2, os, time, json, uuid, traceback, asyncio
import numpy as np
from datetime import datetime, timezone
from typing import Optional, List
from collections import deque
from threading import Lock, Thread
# from pathlib import Path

#------------------------------
from fastapi import FastAPI, UploadFile, File, Form, Query, Path, HTTPException
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

from fastapi.responses import Response
from agno.agent import Agent
from agno.db.postgres import PostgresDb
from agno.models.openai import OpenAIChat
from agno.exceptions import CheckTrigger, InputCheckError
from agno.guardrails import BaseGuardrail, PromptInjectionGuardrail
from agno.run.team import TeamRunInput
from agno.media import Audio
import base64

#---------------------------------
#Interface
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

#----------------------------------
# Accident Detector
from backend.accident_detector import AccidentDetector

# Pothole Detector
from backend.pothole_detector import PotholeDetector

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
DEFAULT_MODEL = os.path.join(PROJECT_ROOT, "models", "accident-yolov11n.pt")
DEFAULT_POTHOLE_MODEL = os.path.join(PROJECT_ROOT, "models", "pothole-yolo.pt")
DEFAULT_OUT   = os.path.join(PROJECT_ROOT, "outputs")

# env vars
MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL)
POTHOLE_MODEL_PATH = os.getenv("POTHOLE_MODEL_PATH", DEFAULT_POTHOLE_MODEL)
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
db_url = DATABASE_URL
api_key_openai  = os.getenv("OPENAI_API_KEY")

# Initialize database connection
db = PostgresDb(db_url=DATABASE_URL)

# Accident Vector DB
knowledge_Accident_reports = Knowledge(
    vector_db=PgVector(
        table_name="Accident_reports",
        db_url=DATABASE_URL,
        embedder=embedder,
        search_type=SearchType.hybrid
    ),
    max_results=5
)

# Pothole Vector DB
knowledge_Pothole = Knowledge(
    vector_db=PgVector(
        table_name="Pothole_report",
        db_url=DATABASE_URL,
        embedder=embedder,
        search_type=SearchType.hybrid
    ),
    max_results=5
)

# Accident Statistics Vector DB
knowledge_Accident_Statistics = Knowledge(
    vector_db=PgVector(
        table_name="Traffic_Accident_Statistics_openai",
        db_url=DATABASE_URL,
        embedder=embedder,
        search_type=SearchType.hybrid
    ),
    max_results=5
)
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

# Request and Response models
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    success: bool

# Guardrail for topic relevance
class TopicRelevanceResult(BaseModel):
    is_relevant: bool
    reason: str

class TrafficIncidentGuardrail(BaseGuardrail):
    """Guardrail to check if query is related to traffic incidents."""
    
    def check(self, run_input: TeamRunInput) -> None:
        if isinstance(run_input.input_content, str):
            validator = Agent(
                model=OpenAIChat(id="gpt-4o-mini", api_key=api_key_openai),
                instructions=[
                    "Determine if the user's question is related to traffic incidents, accidents, road safety, or transportation data.",
                    "Consider synonyms like: mobility, collisions, crashes, road events, transportation incidents.",
                    "Return is_relevant=True if related, False otherwise."
                ],
                output_schema=TopicRelevanceResult,
            )
            
            result = validator.run(
                input=f"Is this question about traffic incidents? '{run_input.input_content}'"
            ).content
            
            if not result.is_relevant:
                raise InputCheckError(
                    f"This system only handles traffic incident queries. {result.reason}",
                    check_trigger=CheckTrigger.OFF_TOPIC,
                )
    
    async def async_check(self, run_input: TeamRunInput) -> None:
        self.check(run_input)

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

    # Get signed URL
    try:
        signed = bucket.create_signed_url(dest_path, 604800) # 7 Days
        # signed is a dict: {'signedURL': '...', 'signedUrl': '...'}
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

def process_confirmed_accident(frame, acc_boxes, camera_id,mean_conf = None, latitude=None, longitude=None):
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

        # Filter (Like if there is false alarms that detected, the vlm will say its not an accident)
        #    If vlm_result["Contact_level"] == "No-contact" or "Unknown"
        #    we will treat it as false alarm -> do NOT upload, do NOT insert DB

        # get signed URL
        dest_path = f"{camera_id}/{event_id}.jpg"
        public_url = upload_image_to_supabase(img_bytes, dest_path)  

        # DB insert
        bbox_list = [bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2], bbox_xyxy[3]]
        time_utc = datetime.now(timezone.utc).isoformat()
        image_uuid = str(uuid.uuid4())

        asyncio.run(db_insert_accident(
            engine, knowledge_Accident_reports,
            event_id=event_id,
            camera_id=camera_id,
            time_utc=time_utc,
            public_url=public_url,
            image_uuid=image_uuid,
            bbox_xyxy=bbox_list,
            mean_conf = mean_conf,
            latitude=latitude,
            longitude=longitude,
            vlm_result=vlm_result
        ))

        print(f"accident: stored event_id={event_id}")

    except Exception as e:
        print("Error in _process_confirmed_accident:", e)
        traceback.print_exc()

def process_confirmed_pothole(frame, pot_boxes, camera_id,mean_conf = None, latitude=None, longitude=None):
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
            mean_conf = mean_conf,
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

# Static (already serving /files). Add a /static for CSS/icons.
app.mount("/static", StaticFiles(directory=os.path.join(PROJECT_ROOT, "..", "static")), name="static")

# Templates
TEMPLATES_DIR = os.path.join(PROJECT_ROOT, "..", "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# >> load models
detector = AccidentDetector(MODEL_PATH, IMG_SIZE)
pothole_model = PotholeDetector(POTHOLE_MODEL_PATH, IMG_SIZE) 

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
# >> Stream control 

def stream_set(camera_id: str, running: bool):
    with STREAM_LOCK:
        STREAM_FLAGS[camera_id] = running

def stream_is_running(camera_id: str) -> bool:
    with STREAM_LOCK:
        return STREAM_FLAGS.get(camera_id, False)

# POST http://localhost:8000/stream/start?camera_id=cam01
@app.post("/stream/start")
def stream_start(camera_id: str = Query("cam01")):
    stream_set(camera_id, True)
    return {"ok": True, "camera_id": camera_id, "running": True}

@app.post("/stream/stop")
# POST http://localhost:8000/stream/stop?camera_id=cam01
def stream_stop(camera_id: str = Query("cam01")):
    stream_set(camera_id, False)
    return {"ok": True, "camera_id": camera_id, "running": False}

@app.get("/stream/status")
#GET http://localhost:8000/stream/status?camera_id=cam01
def stream_status(camera_id: str = Query("cam01")):
    return {"camera_id": camera_id, "running": stream_is_running(camera_id)}

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
        while stream_is_running(camera_id):
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

                if not (same_time or same_roi):
                    #best_conf = float(max(acc, key=lambda d: d[4])[4])
                    avg_conf = float(sum(d[4] for d in acc) / len(acc))
                    _submit_bg(process_confirmed_accident, vis, acc, camera_id, mean_conf=avg_conf)
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
                    if not (same_time_pot or same_roi_pot):
                        #best_conf = float(max(pot, key=lambda d: d[4])[4])
                        avg_conf = float(sum(d[4] for d in pot) / len(pot))
                        _submit_bg(process_confirmed_pothole, vis, pot, camera_id, mean_conf = avg_conf)
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
    stream_set(camera_id, True)
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

@app.get("/events/{event_id}")
def get_event_meta(event_id: str):
    stmt = text("""
        SELECT event_id, camera_id, time_utc, mean_conf, thumbnail_url
        FROM public.incidents
        WHERE event_id = :event_id
        LIMIT 1
    """)
    with engine.connect() as conn:
        row = conn.execute(stmt, {"event_id": event_id}).mappings().first()
    if not row:
        return JSONResponse(status_code=404, content={"detail": "Event not found"})
    return dict(row)

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

@app.get("/potholes/{event_id}")
def get_pothole_meta(event_id: str):
    stmt = text("""
        SELECT event_id, camera_id, time_utc, size, thumbnail_url, bbox_xyxy
        FROM public.potholes WHERE event_id = :event_id LIMIT 1
    """)
    with engine.connect() as conn:
        row = conn.execute(stmt, {"event_id": event_id}).mappings().first()
    if not row:
        return JSONResponse(status_code=404, content={"detail": "Pothole not found"})
    return dict(row)

# --------------------------------------------------------------
# RAG


# Topic router for knowledge base selection
async def topic_router(agent: Agent, query: str, num_documents: int = 5, **kwargs):
    """Route the query to the appropriate knowledge base."""
    if "accident_reports" in query.lower() or "accident reports" in query.lower():
        docs = await knowledge_Accident_reports.async_search(query, max_results=num_documents)
    elif "pothole_report" in query.lower() or "pothole report" in query.lower():
        docs = await knowledge_Pothole.async_search(query, max_results=num_documents)
    elif "accident_statistics" in query.lower() or "accident statistics" in query.lower():
        docs = await knowledge_Accident_Statistics.async_search(query, max_results=num_documents)
    else:
        docs = await knowledge_Accident_reports.async_search(query, max_results=num_documents)
    
    return [d.to_dict() for d in docs]

# Initialize database
db = PostgresDb(db_url=db_url)

# Shared instructions for both agents
AGENT_INSTRUCTIONS = """
You are a RAG Agent specialized in answering questions strictly based on a **local knowledge base** stored in a Vector Database.
There are three available topics, and you must decide which one the user's query belongs to before answering:

- Accident_reports → For questions about specific car accidents or accident reports (e.g., time, location, details of an incident).
- Pothole_report → For questions about road conditions, potholes, or street surface issues.
- Accident_Statistics → For questions about accident numbers, yearly or monthly statistics, or overall accident trends.

Your rules:
1. Always use information ONLY from the local knowledge base (the vector DB). Never use outside or general knowledge.
2. Always identify the correct topic and include it at the start of your answer as: "SelectedTopic: <topic_name>".
3. If relevant information is found, answer clearly and concisely using the retrieved content as evidence.
4. If no relevant information is found in the local knowledge base, respond with an apology such as:
   "Sorry, I could not find any matching information in the local knowledge base."
5. DO NOT generate or guess any information that is not supported by the retrieved local data.
6. Do not add your own information.

When responding via audio, speak clearly and naturally.
"""

# Create TEXT Agent (for text chat)
text_agent = Agent(
    name="Text RAG Agent",
    model=OpenAIChat(id="gpt-4o", api_key=api_key_openai, modalities=["text"]),
    knowledge_retriever=topic_router,
    search_knowledge=True,
    pre_hooks=[PromptInjectionGuardrail(), TrafficIncidentGuardrail()],
    db=db,
    read_chat_history=True,
    add_history_to_context=True,
    num_history_runs=7,
    instructions=AGENT_INSTRUCTIONS,
)

# Create AUDIO Agent (for voice chat)
audio_agent = Agent(
    name="Audio RAG Agent",
    model=OpenAIChat(
        id="gpt-4o-audio-preview",
        modalities=["text", "audio"],
        audio={"voice": "sage", "format": "wav"},
        api_key=api_key_openai
    ),
    knowledge_retriever=topic_router,
    search_knowledge=True,
    #pre_hooks=[PromptInjectionGuardrail(), TrafficIncidentGuardrail()],
    db=db,
    read_chat_history=True,
    add_history_to_context=True,
    num_history_runs=7,
    instructions=AGENT_INSTRUCTIONS,
)

# API Endpoints
@app.get("/api")
async def root():
    """Health check endpoint."""
    return {
        "message": "Unified Traffic Incident RAG API is running",
        "status": "active",
        "endpoints": {
            "text_chat": "/chat",
            "audio_chat": "/audio-chat",
            "health": "/health"
        }
    }

@app.post("/chat", response_model=ChatResponse)
async def text_chat(request: ChatRequest):
    """
    TEXT chat endpoint - processes text queries and returns text responses.
    
    Args:
        request: ChatRequest containing user message
        
    Returns:
        ChatResponse with agent's text response
    """
    try:
        # Run the TEXT agent with user's message
        response = await text_agent.arun(request.message)
        
        # Extract response content
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        return ChatResponse(
            response=response_text,
            success=True
        )
    
    except InputCheckError as e:
        # Handle guardrail violations
        return ChatResponse(
            response=str(e),
            success=False
        )
    
    except Exception as e:
        # Handle other errors
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/audio-chat")
async def voice_chat(audio_file: UploadFile = File(...)):
    """
    AUDIO chat endpoint - processes audio input and returns audio response.
    
    Args:
        audio_file: Audio file from user (WAV format)
        
    Returns:
        Audio response in WAV format
    """
    try:
        # Read audio file content
        audio_content = await audio_file.read()
        
        # Run AUDIO agent with audio input
        response = await audio_agent.arun(
            "Please answer the question in this audio recording based on the knowledge base.",
            audio=[Audio(content=audio_content, format="wav")]
        )
        # Manual parsing of response for audio content
        audio_response_content = None
        
        try:
            response_dict = response.__dict__
            if 'response_audio' in response_dict:
                resp_audio = response_dict['response_audio']
                if resp_audio and hasattr(resp_audio, 'content'):
                    audio_response_content = resp_audio.content
                elif isinstance(resp_audio, dict) and 'content' in resp_audio:
                    audio_response_content = resp_audio['content']
            
            # Check 'audio' field (alternative location)
            if not audio_response_content and 'audio' in response_dict:
                audio_list = response_dict['audio']
                if audio_list and len(audio_list) > 0:
                    if hasattr(audio_list[0], 'content'):
                        audio_response_content = audio_list[0].content
        
        except Exception as parse_error:
            raise HTTPException(
                status_code=500,
                detail=f"Error parsing audio response: {str(parse_error)}"
            )
        
        if not audio_response_content:
            raise HTTPException(
                status_code=500,
                detail="No audio content found in response"
            )
        
        # Decode base64 if needed
        final_audio_bytes = None
        
        try:
            if isinstance(audio_response_content, str):
                print("Decoding base64 audio...")
                final_audio_bytes = base64.b64decode(audio_response_content)
                print(f"Decoded {len(final_audio_bytes)} bytes from base64")
            
            elif isinstance(audio_response_content, bytes):
                if not audio_response_content.startswith(b'RIFF'):
                    print("Attempting to decode base64 from bytes...")
                    try:
                        final_audio_bytes = base64.b64decode(audio_response_content)
                        print(f"Decoded {len(final_audio_bytes)} bytes from base64")
                    except:
                        print("Not base64, using raw bytes")
                        final_audio_bytes = audio_response_content
                else:
                    print("Already valid WAV bytes")
                    final_audio_bytes = audio_response_content
        
        except Exception as decode_error:
            print(f"Decode error: {decode_error}")
            final_audio_bytes = audio_response_content
        
        if not final_audio_bytes or len(final_audio_bytes) == 0:
            raise HTTPException(
                status_code=500,
                detail="Audio decoding resulted in empty content"
            )
        
        print(f"Final audio size: {len(final_audio_bytes)} bytes")
        
        # Fix WAV header if needed
        fixed_audio = final_audio_bytes
        try:
            if final_audio_bytes[:4] == b'RIFF':
                actual_size = len(final_audio_bytes) - 8
                fixed_audio = b'RIFF' + actual_size.to_bytes(4, 'little') + final_audio_bytes[8:]
                print(f"Fixed WAV header")
        except Exception as fix_error:
            print(f"Header fix error: {fix_error}")
        
        return Response(
            content=fixed_audio,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "inline; filename=response.wav",
                "Content-Length": str(len(fixed_audio)),
                "Accept-Ranges": "bytes",
                "Cache-Control": "no-cache"
            }
        )
    
    except InputCheckError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

@app.get("/health")
async def health_check():
    """Check if all services are working."""
    return {
        "api": "healthy",
        "database": "connected",
        "text_agent": "ready",
        "audio_agent": "ready"
    }

# >> UI ROUTES 

@app.get("/home", response_class=HTMLResponse)
def ui_home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/ui/live", response_class=HTMLResponse)
def ui_live(request: Request, camera_id: str = "cam01"):
    return templates.TemplateResponse("live.html", {"request": request, "camera_id": camera_id})

@app.get("/ui/dashboard", response_class=HTMLResponse)
def ui_dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/ui/chat", response_class=HTMLResponse)
def ui_chat(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

#  dashboard cards
@app.get("/partials/events", response_class=HTMLResponse)
def partial_events(request: Request, limit: int = 24):
    # reuse your DB query
    stmt = text("""
        SELECT event_id, camera_id, time_utc, mean_conf, thumbnail_url
        FROM public.incidents ORDER BY time_utc DESC LIMIT :limit
    """)
    with engine.connect() as conn:
        rows = conn.execute(stmt, {"limit": limit}).mappings().all()
    return templates.TemplateResponse("_events.html", {"request": request, "events": rows})

@app.get("/partials/potholes", response_class=HTMLResponse)
def partial_potholes(request: Request, limit: int = 24):
    stmt = text("""
        SELECT event_id, camera_id, time_utc, size, thumbnail_url
        FROM public.potholes ORDER BY time_utc DESC LIMIT :limit
    """)
    with engine.connect() as conn:
        rows = conn.execute(stmt, {"limit": limit}).mappings().all()
    return templates.TemplateResponse("_events.html", {"request": request, "potholes": rows})

@app.get("/ui/event", response_class=HTMLResponse)
def ui_event_details(request: Request):
    # eventId is taken on the client via ?id=...
    return templates.TemplateResponse("event_details.html", {"request": request})

# UI route
@app.get("/ui/pothole", response_class=HTMLResponse)
def ui_pothole_details(request: Request):
    return templates.TemplateResponse("pothole_details.html", {"request": request})