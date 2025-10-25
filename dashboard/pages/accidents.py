import os
import requests
import streamlit as st
import json

# >> Backend URL (FastAPI : localhost:8000)
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# >> Page Setup 
st.set_page_config(page_title="RADS – Accidents", layout="wide")
st.title("Accident Dashboard")

# >> functions
# >> fetch accidents and pothole events
def fetch_events(limit=100):
    try:
        res = requests.get(f"{BACKEND_URL}/events", params={"limit": limit}, timeout=10)
        res.raise_for_status()
        return res.json().get("events", [])
    except Exception as e:
        st.error(f"Error loading events: {e}")
        return []

# >> fetch report for a single event_id
def fetch_report(event_id: str):
    try:
        url = f"{BACKEND_URL}/events/{event_id}/report"
        res = requests.get(url, timeout=30)
        res.raise_for_status()
        return res.json()  # { event_id, report_text, cached }
    except Exception as e:
        st.error(f"Error loading report for {event_id}: {e}")
        return None

# >> Load events 
if "events_cache" not in st.session_state:
    st.session_state["events_cache"] = fetch_events(limit=200)

events = st.session_state.get("events_cache", [])

# >> If Empty   
if not events:
    st.info("No accident events to show yet.")
    st.stop()

# >> Load reports 
if "reports_cache" not in st.session_state:
    st.session_state["reports_cache"] = {}

# >> Pages
per_page = 6 # num of cards in page
page = st.session_state.get("page", 1)
total = len(events)
pages = max(1, (total + per_page - 1) // per_page)
colA, colB, colC = st.columns([1, 2, 1])
with colA:
    if st.button("<- Prev", disabled=(page <= 1)):
        page = max(1, page - 1)
with colB:
    st.markdown(f"<div style='text-align:center;'>Page {page} / {pages}</div>", unsafe_allow_html=True)
with colC:
    if st.button("Next ->", disabled=(page >= pages)):
        page = min(pages, page + 1)
st.session_state["page"] = page

start, end = (page - 1) * per_page, (page - 1) * per_page + per_page
page_events = events[start:end]

# >> Images Grid  
cols_per_row = 3

for i in range(0, len(page_events), cols_per_row):
    row = st.columns(cols_per_row)
    for j, ev in enumerate(page_events[i:i + cols_per_row]):
        with row[j]:
            event_id = ev.get("event_id","")
            cam = ev.get("camera_id","")
            utc = ev.get("time_utc","")
            conf = float(ev.get("mean_conf", 0.0))
            thumb_rel = ev.get("thumbnail_url")

            # header (camera ID + time)
            st.markdown(f"<div style='font-weight:700;font-size:1.05rem;'>{cam}</div>", unsafe_allow_html=True)
            st.caption(f"UTC: {utc}")
            st.caption(f"Confidence: {conf:.2f}")

            # show thumbnail
            if thumb_rel:
                st.image(f"{BACKEND_URL}{thumb_rel}", use_container_width=True)

            # accident id
            st.caption(f"Accident ID: {event_id}")

            # report
            if st.button("Show report", key=f"show_{event_id}"):
                with st.spinner("Generating / loading report..."):
                        # serve from local session cache 
                        cached = st.session_state["reports_cache"].get(event_id)
                        if cached is None:
                            resp = fetch_report(event_id)
                            if resp:
                                st.session_state["reports_cache"][event_id] = resp
                                cached = resp
                        if cached:
                            # a JSON string in report_text
                            report_text = cached.get("report_text", "")
                            try:
                                parsed = json.loads(report_text)
                                st.json(parsed, expanded=False)
                            except Exception:
                                # not JSON > just show text
                                st.text_area("Report", report_text, height=220)
