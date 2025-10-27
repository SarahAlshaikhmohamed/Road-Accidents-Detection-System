import os, json, requests, pathlib, html
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from components.ui import render_sidebar

# >> Backend URL (FastAPI : localhost:8000)
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
st.set_page_config(page_title="RADS – Accident Details", layout="wide")
st.markdown("<h2 style='margin:0;'>Event details</h2>", unsafe_allow_html=True)

# sidebar
render_sidebar() 

# >> Load CSS
css_path = pathlib.Path(__file__).parent / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] .main .block-container{padding-top:.5rem}
    .rads-card{background:linear-gradient(180deg, #232323, #2C2C2C);border:1px solid #3A3A3A;
               border-radius:18px;box-shadow:0 8px 28px rgba(0,0,0,.35);overflow:hidden}
    .rads-pad{padding:16px 18px}
    .image-shell{position:relative;border:1px solid #3A3A3A;border-radius:16px;overflow:hidden}
    .ribbon{position:absolute;top:12px;left:12px;z-index:2;padding:6px 12px;border-radius:12px;
            color:#fff;font-weight:800;font-size:.92rem;box-shadow:0 8px 28px rgba(0,0,0,.35)}
    .kv{display:flex;gap:10px;margin:6px 0;align-items:flex-start;
        border-bottom:1px dashed rgba(255,255,255,.08);padding-bottom:6px}
    .kv .k{min-width:200px;opacity:.85;font-weight:800}
    .kv .v{flex:1}
    .hr{height:1px;background:#3A3A3A;border:0;margin:12px 0}
    .meta-chip{display:inline-block;padding:6px 10px;border-radius:10px;
               background:rgba(255,255,255,.06);border:1px solid #3A3A3A;font-weight:700}
    </style>
    """, unsafe_allow_html=True)

# >> Helper functions
SEV_COLORS = {
    "Low":     "#facc15",
    "Medium":  "#fb923c",
    "High":    "#ef4444",
    "Unknown": "#9ca3af",
}
BLUE_FALSE_ALARM = "#3b82f6"

def is_false_alarm(contact_level: str | None) -> bool:
    return (contact_level or "").lower() in {"Unknown", "No-contact"}

def sev_color(severity: str | None) -> str:
    return SEV_COLORS.get((severity or "Unknown").title(), SEV_COLORS["Unknown"])
def kv_line(label: str, value):
    value = "-" if value in (None, "", []) else value
    st.markdown(
        f"<div style='display:flex;gap:8px;margin:4px 0;'>"
        f"<div style='min-width:180px;opacity:.8;font-weight:700;'>{label}:</div>"
        f"<div>{value}</div>"
        f"</div>",
        unsafe_allow_html=True
    )

# >> Get event meta
meta = st.session_state.get("event_details") or {}
event_id = meta.get("event_id")
thumb    = meta.get("thumbnail_url")
camera   = meta.get("camera_id")
utc_iso  = meta.get("time_utc")

# >> Header info
cols1, cols2 = st.columns([1, 1])
with cols1:
    st.markdown(
        f"<span class='meta-chip'><b>Event:</b> <code>{html.escape(str(event_id)) if event_id else '-'}</code></span>",
        unsafe_allow_html=True
    )
    st.caption(f"Camera: {camera or '-'} • UTC: {utc_iso or '-'}")
with cols2:
    st.markdown("<div style='text-align:right;'>", unsafe_allow_html=True)
    if st.button("< Back to Dashboard"):
        switch_page("accidents")
    st.markdown("</div>", unsafe_allow_html=True)

# Fetch Report
r = requests.get(f"{BACKEND_URL}/events/{event_id}/report", timeout=30)
r.raise_for_status()
report_payload = r.json()
report = report_payload.get("report") or {}

# Pull info
category        = report.get("Category")
contact_level   = report.get("Contact_level")
derivation      = report.get("Derivation_object")
environment     = report.get("Environment")
time_of_day     = report.get("Time")
lane            = report.get("Traffic_lane_of_the_object")
weather         = report.get("Weather")
severity        = report.get("Severity")
emergency       = report.get("Emergency_lights")
vehicles_count  = report.get("Vehicles_count")
evidence_list   = report.get("Evidence") or []
conf            = report.get("Confidence") or {}

# color + label
if is_false_alarm(contact_level):
    ribbon_text  = "False Alarm"
    ribbon_color = BLUE_FALSE_ALARM
else:
    ribbon_text  = f"Severity: {severity or 'Unknown'}"
    ribbon_color = sev_color(severity)

# Event details
left, right = st.columns([6, 10], gap="large")
# >> Left card (image)
with left:
    st.markdown("<div class='image-shell'>", unsafe_allow_html=True)
    st.markdown(f"<span class='ribbon' style='background:{ribbon_color};'>{html.escape(ribbon_text)}</span>", unsafe_allow_html=True)
    if thumb:
        st.image(thumb, use_column_width=True)
    else:
        st.caption("No image available.")
    st.markdown("</div>", unsafe_allow_html=True)

# >> Right card
with right:
    #st.markdown("<div class='rads-card rads-pad'>", unsafe_allow_html=True)

    st.subheader("Overview")
    kv_line("Category", category)
    kv_line("Contact level", contact_level)
    kv_line("Derivation object", derivation)
    kv_line("Vehicles count", vehicles_count)
    kv_line("Environment", environment)
    kv_line("Weather", weather)
    kv_line("Time of day", time_of_day)
    kv_line("Traffic lane", lane)
    kv_line("Emergency lights", emergency)

    # >> Evidence
    if evidence_list:
        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
        st.markdown("<div style='font-weight:800;'>Evidence</div>", unsafe_allow_html=True)
        safe_items = "".join(f"<li>{html.escape(str(e))}</li>" for e in evidence_list)
        st.markdown(f"<ul class='ev'>{safe_items}</ul>", unsafe_allow_html=True)

    # >> Confidences
    with st.expander("Confidences"):
        if conf:
            for k, v in conf.items():
                kv_line(str(k), v)
        else:
            st.caption("No confidence scores available.")

    # >> Actions
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    a1, a2, _ = st.columns([1, 1, 2])
    with a1:
        st.download_button(
            "Download JSON report",
            data=json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=f"{(event_id or 'event')}_report.json",
            mime="application/json",
            use_container_width=True,
        )
    with a2:
        if thumb:
            st.link_button("Open image", url=thumb, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)