import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import streamlit_antd_components as sac
from streamlit_lottie import st_lottie
import json, pathlib, requests

st.set_page_config(page_title="RADS: Road Accident Detection System", layout="wide")

# loading css file
css_path = pathlib.Path(__file__).parent / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# to hide streamlit default sidebar and shrink the header 
st.markdown("""
<style>
[data-testid="stSidebarNav"]{display:none;}
header[data-testid="stHeader"]{height:12px;padding:0;box-shadow:none;background:transparent;}
main .block-container{padding-top:.5rem;}
</style>
""", unsafe_allow_html=True)

# Top menu 
choice = sac.menu(
    items=[
        sac.MenuItem('Home', icon='house-fill'),
        sac.MenuItem('Overview', icon='bar-chart-line-fill'),
        sac.MenuItem('Accidents', icon='grid-3x3-gap-fill'),
        sac.MenuItem('Live Stream', icon='camera-video-fill'),
        sac.MenuItem('Chat', icon='chat-dots-fill'),
    ],
    index=0,
    color='#004D61'
)

if choice == 'Overview':
    switch_page('eda')
elif choice == 'Accidents':
    switch_page('accidents')
elif choice == 'Live Stream':
    switch_page('live stream')
elif choice == 'Chat':
    switch_page('chat')

# 
st.markdown("""
<div class="rads-hero-glass">
  <h1><b>Welcome to RADS</b></h1>
  <p>AI-powered accident & pothole detection for <b>smart cities</b>.</p>
</div>
""", unsafe_allow_html=True)

# Lottie
def load_lottie(url_or_path: str):
    r = requests.get(url_or_path, timeout=8)
    if r.ok:
        return r.json()

lottie = load_lottie("https://lottie.host/1b2e4c72-87cf-4cdf-a027-131ee5d00b55/1aw3EuQKSP.json")
if lottie:
    st_lottie(lottie, height=440, key="car_anim")



# >> Feature cards 
st.markdown("<h2 class='center-title'>Explore Features</h2>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown("""
    <div class="rads-feature-card card-teal" onclick="window.location.href='live%20stream'">
        <div>
          <div class="rads-feature-title">Live Detection</div>
          <div class="rads-feature-desc">Description</div>
        </div>
        <div class="rads-feature-cta">Open</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open", key="open_live", use_container_width=True):
        switch_page("live stream")

with col2:
    st.markdown("""
    <div class="rads-feature-card card-violet" onclick="window.location.href='accidents'">
        <div>
          <div class="rads-feature-title">Dashboard</div>
          <div class="rads-feature-desc">Description</div>
        </div>
        <div class="rads-feature-cta">Open</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open", key="open_dash", use_container_width=True):
        switch_page("accidents")

with col3:
    st.markdown("""
    <div class="rads-feature-card card-ruby" onclick="window.location.href='chat'">
        <div>
          <div class="rads-feature-title">Chatbot</div>
          <div class="rads-feature-desc"></div>
        </div>
        <div class="rads-feature-cta">Open</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open", key="open_chat", use_container_width=True):
        switch_page("chat")
