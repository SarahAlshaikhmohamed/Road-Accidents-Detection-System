import os, requests
import streamlit as st
import streamlit_antd_components as sac
from streamlit_extras.switch_page_button import switch_page
from components.ui import render_sidebar

# >> Backend URL (FastAPI : localhost:8000)
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# >> Page Setup 
st.set_page_config(page_title="RADS Dashboard", layout="wide")
st.title("RADS Dashboard")

# sidebar
render_sidebar() 

# >> functions helper 

# >> load recent accident events from FastAPI /events
def fetch_events(limit: int = 100):
    r = requests.get(f"{BACKEND_URL}/events", params={"limit": limit}, timeout=10)
    r.raise_for_status()
    return r.json().get("events", [])

# >> fetch Potholes data from FastAPI /potholes
def fetch_potholes(limit: int = 100):
    r = requests.get(f"{BACKEND_URL}/potholes", params={"limit": limit}, timeout=10)
    r.raise_for_status()
    return r.json().get("potholes", [])

# -----------------------------
# UI Design
SEV_COLORS = {
    "Low":     "#facc15",  # yellow
    "Medium":  "#fb923c",  # orange
    "High":    "#ef4444",  # red
    "Unknown": "#9ca3af",  # gray
}
BLUE_FALSE_ALARM = "#3b82f6"  # blue if Contact level either : Unknown or No-contact

def ribbon_style(color: str) -> str:
    return (
        f"position:absolute;top:10px;left:10px;background:{color};"
        "color:#fff;font-weight:800;font-size:.85rem;"
        "padding:4px 10px;border-radius:10px;"
        "box-shadow:0 6px 22px rgba(0,0,0,.35);"
    )

# Unknown or No-contact > Blue > False alarm
def is_false_alarm(contact_level: str | None) -> bool:
    return (contact_level or "").lower() in {"unknown", "no-contact", "no contact"}

def sev_color(severity: str | None) -> str:
    return SEV_COLORS.get((severity or "Unknown").title(), SEV_COLORS["Unknown"])

def card_html(*, image_url: str | None, camera_id: str, utc_iso: str, event_id: str, badge_text: str,badge_color: str, ):
    img = image_url or ""
    return f"""
    <div style="
        position:relative;overflow:hidden;
        background: linear-gradient(180deg, var(--surface), var(--surface-2));
        border:1px solid var(--border); border-radius: var(--radius);
        box-shadow: var(--shadow); height:100%; display:flex; flex-direction:column;">
        <span style="{ribbon_style(badge_color)}">{badge_text}</span>
        <div style="aspect-ratio:16/10;width:100%;overflow:hidden;background:#111;">
            <img src="{img}" style="width:100%;height:100%;object-fit:cover;display:block;" />
        </div>
        <div style="padding:12px 14px 14px;">
            <div style="display:flex;justify-content:space-between;gap:8px;">
                <div style="font-weight:800;">{camera_id}</div>
                <div style="opacity:.7;font-size:.9rem;">{utc_iso}</div>
            </div>
            <div style="opacity:.75;font-size:.9rem;margin-top:4px;">Event: {event_id}</div>
        </div>
    </div>
    """

#----------------------------------------------
# Data 
if "acc_limit" not in st.session_state:
    st.session_state.acc_limit = 12  # first 12 accidents
if "poth_limit" not in st.session_state:
    st.session_state.poth_limit = 12 # first 12 potholes

# fetch
ACCIDENTS = fetch_events(limit=200)
POTHOLES  = fetch_potholes(limit=200)

if not ACCIDENTS and not POTHOLES:
    st.info("No events yet.")
    st.stop()

# Tabs
active = sac.tabs(
    items=[
        sac.TabsItem(label='Accidents', icon='alert'),
        sac.TabsItem(label='Potholes',  icon='tool'),
    ],
    align='center',
    color='rgb(20,80,90)',
    key="main_tabs"
)

# -----------------------------
# ACCIDENTS TAB
if active == "Accidents":
    shown = ACCIDENTS[: st.session_state.acc_limit]
    if not shown:
        st.info("No accident events to show yet.")
    else:
        cols = st.columns(3, gap="large")
        for idx, ev in enumerate(shown):
            with cols[idx % 3]:
                event_id = ev.get("event_id", "")
                cam      = ev.get("camera_id", "")
                utc      = ev.get("time_utc", "")
                thumb    = ev.get("thumbnail_url")

                #  false-alarm (blue) OR severity color 
                # For the list page, we show a neutral "Accident" badge; the event page will render the real severity/false-alarm.
                badge_text = "Accident"
                badge_color = "#595959"  # neutral at list level

                st.markdown(card_html(
                    image_url=thumb, camera_id=cam, utc_iso=utc,
                    event_id=event_id, badge_text=badge_text, badge_color=badge_color
                ), unsafe_allow_html=True)

                # View details 
                if st.button("View details", key=f"view_{event_id}", use_container_width=True):
                    st.session_state["event_details"] = {
                        "event_id": event_id,
                        "thumbnail_url": thumb,
                        "camera_id": cam,
                        "time_utc": utc,
                    }
                    switch_page("event details")

        # Load more (+12)
        if len(ACCIDENTS) > st.session_state.acc_limit:
            if st.button("Load more", use_container_width=True):
                st.session_state.acc_limit += 12

# -----------------------------
# POTHOLES TAB
elif active == "Potholes":
    shown = POTHOLES[: st.session_state.poth_limit]
    if not shown:
        st.info("No potholes to show yet.")
    else:
        cols = st.columns(3, gap="large")
        for idx, ev in enumerate(shown):
            with cols[idx % 3]:
                event_id = ev.get("event_id", "")
                cam      = ev.get("camera_id", "")
                utc      = ev.get("time_utc", "")
                size     = ev.get("size", "-")
                thumb    = ev.get("thumbnail_url")

                st.markdown(card_html(
                    image_url=thumb, camera_id=cam, utc_iso=utc,
                    event_id=event_id, badge_text=f"Pothole: {size or '-'}",
                    badge_color="#0ea5e9"  # cyan-ish
                ), unsafe_allow_html=True)

                if st.button("View details", key=f"view_p_{event_id}", use_container_width=True):
                    st.session_state["event_details"] = {
                        "event_id": event_id,
                        "thumbnail_url": thumb,
                        "camera_id": cam,
                        "time_utc": utc,
                        "kind": "pothole",
                        "size": size,
                    }
                    switch_page("event details")

        # Load more
        if len(POTHOLES) > st.session_state.poth_limit:
            if st.button("Load more", use_container_width=True):
                st.session_state.poth_limit += 12
