import os, requests, streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
st.set_page_config(page_title="Live Stream Control", layout="wide")
st.title("Live stream control")

#cam_id = st.text_input("Camera ID", value="cam01")
cam_id = "cam01"
source = 0

col1, col2, col3 = st.columns([1,1,1])
with col1:
    if st.button("Start stream"):
        try:
            r = requests.post(f"{BACKEND_URL}/stream/start", params={"camera_id": cam_id, "source": source}, timeout=5)
            r.raise_for_status()
            st.success("Stream flag set to RUNNING")
        except Exception as e:
            st.error(f"Start failed: {e}")

with col2:
    if st.button("Stop stream"):
        try:
            r = requests.post(f"{BACKEND_URL}/stream/stop", params={"camera_id": cam_id}, timeout=5)
            r.raise_for_status()
            st.success("Stream flag set to STOPPED")
        except Exception as e:
            st.error(f"Stop failed: {e}")

with col3:
    if st.button("Open live view"):
        st.markdown(f"[Open stream in new tab]({BACKEND_URL}/stream?camera_id={cam_id}&source={source})")

#st.divider()
#st.subheader("Status")
#try:
#    r = requests.get(f"{BACKEND_URL}/stream/status", params={"camera_id": cam_id}, timeout=5)
#    r.raise_for_status()
#    st.json(r.json())
#except Exception as e:
#    st.error(f"Status failed: {e}")

st.divider()
st.subheader("Recent accident events")
try:
    r = requests.get(f"{BACKEND_URL}/events?limit=12", timeout=5)
    r.raise_for_status()
    events = r.json().get("events", [])
    if not events:
        st.info("No events yet")
    else:
        cols_per_row = 3
        for i in range(0, len(events), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, ev in enumerate(events[i:i+cols_per_row]):
                with cols[j]:
                    st.caption(f"{ev['camera_id']} · {ev['time_utc']} · {ev['mean_conf']:.2f}")
                    thumb = ev.get("thumbnail_url")
                    if thumb:
                        st.image(f"{BACKEND_URL}{thumb}", use_container_width=True)
except Exception as e:
    st.error(f"Load events failed: {e}")