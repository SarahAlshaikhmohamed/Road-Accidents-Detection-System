import os, requests, streamlit as st
import streamlit_antd_components as sac
from components.ui import render_sidebar
from streamlit_extras.switch_page_button import switch_page
import streamlit_antd_components as sac

# Backend URL 
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
cam_id = "cam01"
source = "0"
max_fps = 10
conf = 0.75


# >> Page Setup
st.set_page_config(page_title="Live Stream Control", layout="wide")
st.markdown("<h2 style='margin:0;'>Live Stream</h2>", unsafe_allow_html=True)

# sidebar
render_sidebar() 



# >> state
if "running" not in st.session_state:
    st.session_state["running"] = False

# Buttons
clicked = sac.buttons(
    items=[
        sac.ButtonsItem(label='Start', icon='play-fill', color='#52c41a'),  
        sac.ButtonsItem(label='Stop',  icon='stop-fill', color='#ff4d4f'),  
    ],
    align='center',
    key="stream_controls"
)


def _do_start():
    r = requests.post(f"{BACKEND_URL}/stream/start", params={"camera_id": cam_id}, timeout=10)
    r.raise_for_status()
    st.session_state["running"] = True
    st.success("Stream flag set to RUNNING")

def _do_stop():
    r = requests.post(f"{BACKEND_URL}/stream/stop", params={"camera_id": cam_id}, timeout=10)
    r.raise_for_status()
    st.session_state["running"] = False
    st.success("Stream flag set to STOPPED")

# >> Map index
# if 0 > start, 1> stop , neither > None
try:
    if clicked in (0, "Start"):
        _do_start()
    elif clicked in (1, "Stop"):
        _do_stop()
except Exception as e:
    st.error(f"Load events failed: {e}")


st.markdown("<hr style='margin:0.75rem 0;'>", unsafe_allow_html=True)

# Live view
if st.session_state["running"]:
    # /stream?camera_id=cam01&source=0&max_fps=10
    stream_url = (
        f"{BACKEND_URL}/stream"
        f"?camera_id={cam_id}"
        f"&source={source}"
        f"&max_fps={max_fps}"
        f"&conf_acc={conf}"
    )
    #st.write("stream url:", stream_url)
    st.markdown(
        f"""
        <div style="border:1px solid #ddd;border-radius:8px;overflow:hidden;">
            <img src="{stream_url}" style="width:100%;height:auto;" />
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.info("Stream is not running.")