import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from streamlit_lottie import st_lottie
import json, pathlib, requests
from components.ui import render_sidebar

st.set_page_config(page_title="Buad: Smart Road Monitoring System", layout="wide")

# loading css file
css_path = pathlib.Path(__file__).parent / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# >> sidebar
render_sidebar() 

# >> hero
st.markdown("""
<div class="buad-hero-glass">
  <h1><b>Welcome to Buad</b></h1>
  <p>Smart Road Monitoring System — real-time incident detection, pothole analytics, and data-driven insights for modern cities.</p>
</div>
""", unsafe_allow_html=True)

# >> project intro (centered)
st.markdown("""
<div style="max-width:900px;margin:10px auto 0;text-align:center;color:var(--text-muted);line-height:1.6;">
  Buad unifies live video analytics, event dashboards, and a retrieval-augmented assistant.
  Monitor incidents as they happen, explore historical trends, and generate reports — all in one place.
</div>
""", unsafe_allow_html=True)

# >> lottie (optional visual)
def load_lottie(url_or_path: str):
    r = requests.get(url_or_path, timeout=8)
    if r.ok:
        return r.json()

lottie = load_lottie("https://lottie.host/1b2e4c72-87cf-4cdf-a027-131ee5d00b55/1aw3EuQKSP.json")
if lottie:
    st_lottie(lottie, height=440, key="car_anim")

# >> feature cards
st.markdown("<h2 class='center-title'>Explore Features</h2>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown("""
    <div class="buad-feature-card card-teal" onclick="window.location.href='live%20stream'">
        <div>
          <div class="buad-feature-title">Live Detection</div>
          <div class="buad-feature-desc">Real-time video analytics, incident flags, and evidence snapshots from connected cameras.</div>
        </div>
        <div class="buad-feature-cta">Open</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open", key="open_live", use_container_width=True):
        switch_page("live stream")

with col2:
    st.markdown("""
    <div class="buad-feature-card card-violet" onclick="window.location.href='accidents'">
        <div>
          <div class="buad-feature-title">Dashboard</div>
          <div class="buad-feature-desc">Browse incidents, filter by time and location, and view maps, trends, and KPIs.</div>
        </div>
        <div class="buad-feature-cta">Open</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open", key="open_dash", use_container_width=True):
        switch_page("accidents")

with col3:
    st.markdown("""
    <div class="buad-feature-card card-ruby" onclick="window.location.href='chat'">
        <div>
          <div class="buad-feature-title">Chatbot</div>
          <div class="buad-feature-desc">RAG assistant to query accident reports, pothole records, and statistics using natural language.</div>
        </div>
        <div class="buad-feature-cta">Open</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open", key="open_chat", use_container_width=True):
        switch_page("chat")

# >>  footer tagline ]
st.markdown("""
<div style="text-align:center;opacity:.75;margin-top:24px;">
  <small>Buad • Smart Road Monitoring System</small>
</div>
""", unsafe_allow_html=True)