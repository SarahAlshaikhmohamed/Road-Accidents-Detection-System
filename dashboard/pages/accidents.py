import os, requests, datetime as dt
import streamlit as st
import streamlit_antd_components as sac
from streamlit_extras.switch_page_button import switch_page
from components.ui import render_sidebar

# >> Backend URL (FastAPI : localhost:8000)
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# >> Page Setup 
st.set_page_config(page_title="Buad Dashboard", layout="wide")
st.title("Buad Dashboard")

# sidebar
render_sidebar() 

# >> helpers 
@st.cache_data(ttl=60, show_spinner= False)
def fetch_events(limit: int = 100):
    try:
        r = requests.get(f"{BACKEND_URL}/events", params={"limit": limit}, timeout=10)
        r.raise_for_status()
        return r.json().get("events", [])
    except Exception:
        return []

@st.cache_data(ttl=60, show_spinner= False)
def fetch_potholes(limit: int = 100):
    try:
        r = requests.get(f"{BACKEND_URL}/potholes", params={"limit": limit}, timeout=10)
        r.raise_for_status()
        return r.json().get("potholes", [])
    except Exception:
        return []

@st.cache_data(ttl=120, show_spinner= False)
def fetch_report(event_id: str):
    try:
        r = requests.get(f"{BACKEND_URL}/events/{event_id}/report", timeout=10)
        if r.status_code == 200:
            return (r.json() or {}).get("report", {}) or {}
    except Exception:
        pass
    return {}

# -----------------------------
# UI Design
SEV_COLORS = {
    "Low":     "#facc15",  # yellow
    "Medium":  "#fb923c",  # orange
    "High":    "#ef4444",  # red
    "Unknown": "#9ca3af",  # gray
}
BLUE_FALSE_ALARM = "#3b82f6"

def ribbon_style(color: str) -> str:
    return (
        f"position:absolute;top:10px;left:10px;background:{color};"
        "color:#fff;font-weight:800;font-size:.85rem;"
        "padding:4px 10px;border-radius:10px;"
        "box-shadow:0 6px 22px rgba(0,0,0,.35);"
    )

# >> exact wording (no casing changes) per spec
def is_false_alarm(contact_level: str | None) -> bool:
    return (contact_level or "").strip() in {"No-contact", "Unknown"}

def sev_color(severity: str | None) -> str:
    return SEV_COLORS.get((severity or "Unknown").title(), SEV_COLORS["Unknown"])

def card_html(*, image_url: str | None, camera_id: str, utc_iso: str, event_id: str, badge_text: str, badge_color: str):
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
ACCIDENTS = fetch_events(limit=300)
POTHOLES  = fetch_potholes(limit=300)

if not ACCIDENTS and not POTHOLES:
    st.info("No events yet.")
    st.stop()

# >> shared filters (applies to both tabs)
st.markdown("### Filters")
fc1, fc2 = st.columns(2)
with fc1:
    date_from = st.date_input("From", value=None, key="f_from")
with fc2:
    date_to = st.date_input("To", value=None, key="f_to")
#with fc3:
#    camera_search = st.text_input("Camera contains", value="", key="f_cam").strip()

def within_date(iso_str: str) -> bool:
    if not iso_str:
        return True
    try:
        d = dt.datetime.fromisoformat(iso_str.replace("Z","")).date()
    except Exception:
        return True
    if date_from and d < date_from: return False
    if date_to and d > date_to: return False
    return True

#def match_camera(cam: str) -> bool:
#    if not camera_search: return True
#    return camera_search.lower() in str(cam or "").lower()

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
    # >> accidents-only filters
    st.markdown("#### Accident Filters")
    c1, c2= st.columns(2)
    with c1:
        #show_false = st.selectbox("False alarms", ["All", "Exclude", "Only"], index=0, key="fa_mode")
        show_false = st.selectbox("False alarms", ["All", "Exclude", "Only"], index=0, key="fa_mode")
    with c2:
        sev_pick = st.multiselect("Severity", options=["High","Medium","Low","Unknown"], default=[], key="sev_pick")
    #with c3:
    #    conf_min = st.slider("Min. mean confidence", 0.0, 1.0, 0.0, 0.01, key="acc_conf_min")
    #with c4:
    #    conf_max = st.slider("Max. mean confidence", 0.0, 1.0, 1.0, 0.01, key="acc_conf_max")

    # >> apply shared filters first
    base = [e for e in ACCIDENTS if within_date(e.get("time_utc","")) ] #and match_camera(e.get("camera_id",""))]

    # >> enrich each event with VLM fields (severity/contact) from per-event report (cached)
    # keep this lazy and only for shown slice for performance (after filtering by confidence)
    #def conf_ok(e):
    #    try:
    #        m = float(e.get("mean_conf", 0.0))
    #        return (m >= conf_min) and (m <= conf_max)
    #    except Exception:
    #        return True

    #base = [e for e in base if conf_ok(e)]

    # >> peek VLM fields for filtering + badge (cached call per event)
    enriched = []
    for e in base:
        rep = fetch_report(e.get("event_id","")) or {}
        e["_Severity"] = (rep.get("Severity") or "Unknown")
        e["_Contact"]  = (rep.get("Contact_level") or "Unknown")
        enriched.append(e)

    # >> filter by false alarms
    def fa_mode_ok(contact: str) -> bool:
        fa = is_false_alarm(contact)
        if show_false == "All": return True
        if show_false == "Exclude": return not fa
        if show_false == "Only": return fa
        return True

    # >> filter by severity list (if any picked)
    def sev_ok(sev: str) -> bool:
        if not sev_pick: return True
        return (sev or "Unknown") in set(sev_pick)

    filtered = [e for e in enriched if fa_mode_ok(e.get("_Contact")) and sev_ok(e.get("_Severity"))]

    # >> show cards
    shown = filtered[: st.session_state.acc_limit]
    if not shown:
        st.info("No accident events to show with current filters.")
    else:
        cols = st.columns(3, gap="large")
        for idx, ev in enumerate(shown):
            with cols[idx % 3]:
                event_id = ev.get("event_id", "")
                cam      = ev.get("camera_id", "")
                utc      = ev.get("time_utc", "")
                thumb    = ev.get("thumbnail_url")
                sev      = ev.get("_Severity") or "Unknown"
                contact  = ev.get("_Contact")  or "Unknown"

                # >> badge text/color (match event_details)
                if is_false_alarm(contact):
                    badge_text  = "False Alarm"
                    badge_color = BLUE_FALSE_ALARM
                else:
                    badge_text  = f"Severity: {sev}"
                    badge_color = sev_color(sev)

                st.markdown(card_html(
                    image_url=thumb, camera_id=cam, utc_iso=utc,
                    event_id=event_id, badge_text=badge_text, badge_color=badge_color
                ), unsafe_allow_html=True)

                # >> View details 
                if st.button("View details", key=f"view_{event_id}", use_container_width=True):
                    st.session_state["event_details"] = {
                        "event_id": event_id,
                        "thumbnail_url": thumb,
                        "camera_id": cam,
                        "time_utc": utc,
                    }
                    switch_page("event details")

        # >> Load more (+12)
        if len(filtered) > st.session_state.acc_limit:
            if st.button("Load more", use_container_width=True):
                st.session_state.acc_limit += 12

# -----------------------------
# POTHOLES TAB
elif active == "Potholes":
    # >> potholes-only filters
    st.markdown("#### Pothole Filters")
    pc1 = st.columns(1)
    with pc1:
        size_pick = st.multiselect("Size", options=["small","medium","large"], default=[], key="size_pick")
    #with pc2:
    #    cam2 = st.text_input("Camera contains (potholes)", value=camera_search, key="p_cam").strip()

    # >> apply shared + tab filters
    base = [p for p in POTHOLES if within_date(p.get("time_utc","")) ]#and match_camera(cam2 or p.get("camera_id",""))]

    def size_ok(s: str):
        if not size_pick: return True
        return (str(s or "").lower() in {x.lower() for x in size_pick})

    filtered = [p for p in base if size_ok(p.get("size",""))]

    shown = filtered[: st.session_state.poth_limit]
    if not shown:
        st.info("No potholes to show with current filters.")
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
                    badge_color="#0ea5e9"
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

        # >> Load more
        if len(filtered) > st.session_state.poth_limit:
            if st.button("Load more", use_container_width=True):
                st.session_state.poth_limit += 12