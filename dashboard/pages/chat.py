import streamlit as st
import io, base64, requests
import pathlib
from audio_recorder_streamlit import audio_recorder
from components.ui import render_sidebar

# Page configuration
st.set_page_config(
    page_title="Traffic RAG Assistant",
    page_icon=":car:",
    layout="wide"
)

# sidebar
render_sidebar() 

# API endpoint configuration
TEXT_API_URL = "http://localhost:8000/chat"
AUDIO_API_URL = "http://localhost:8000/audio-chat"
HEALTH_URL = "http://localhost:8000/health"

# >> Load css
ROOT = pathlib.Path(__file__).resolve().parents[2]
css_path =  ROOT / "dashboard" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# --------
st.markdown("""
<style>
/* page width wrapper */
.rads-wrap{max-width:1200px;margin:0 auto;}
/* hero card */
.rads-hero{
  margin:12px 0 10px; padding:20px 22px;
  border-radius:var(--radius); border:1px solid var(--border);
  background:linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.02));
  box-shadow:var(--shadow);
}
.rads-hero h1{margin:0 0 6px; font-size:1.9rem; font-weight:900; color:var(--text);}
.rads-hero p {margin:0; color:var(--text-muted);}

/* generic card + padding */
.rads-card{background:linear-gradient(180deg, var(--surface), var(--surface-2));
  border:1px solid var(--border); border-radius:var(--radius); box-shadow:var(--shadow);}
.rads-pad{padding:16px 18px;}

/* mode pill */
.rads-pill{
  display:inline-flex; gap:10px; align-items:center;
  padding:10px 14px; border-radius:999px; border:1px solid var(--border);
  background:var(--surface-2); color:var(--text); font-weight:800;
}

/* bubbles for chat history */
.rads-bubbles{display:flex; flex-direction:column; gap:10px;}
.rads-bubble{
  max-width:900px; padding:12px 14px; border-radius:16px;
  border:1px solid var(--border); box-shadow:var(--shadow);
  white-space:pre-wrap; line-height:1.55;
}
.rads-bubble.user{align-self:flex-end; background:rgba(0,199,165,.10);}
.rads-bubble.assistant{align-self:flex-start; background:rgba(255,255,255,.05);}

/* titles */
.rads-h2{font-size:1.15rem; font-weight:900; margin:0 0 10px;}

/* tidy sidebar text */
.sidebar-muted{color:var(--text-muted); font-size:.95rem;}
/* horizontal rule soft */
.hr{height:1px;background:var(--border);border:0;margin:12px 0;}
</style>
""", unsafe_allow_html=True)

# App title 
st.markdown('<div class="rads-wrap">', unsafe_allow_html=True)
st.markdown("""
<div class="rads-hero">
  <h1>Traffic Incident RAG Assistant</h1>
  <p><b>Unified System</b> - Ask questions using Text or Voice about:</p>
  <ul style="margin:.4rem 0 0 1.1rem; color:var(--text-muted);">
    <li><b>Accident Reports</b> - Specific accident incidents</li>
    <li><b>Pothole Reports</b> - Road conditions</li>
    <li><b>Accident Statistics</b> - Statistical data and trends</li>
  </ul>
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Information")

    # Mode selector in sidebar 
    st.subheader("Select Mode")
    chat_mode = st.radio(
        "Choose interaction mode:",
        ["Text Chat", "Voice Chat"],  
        help="Switch between text typing and voice recording"
    )

    st.markdown("---")
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown("### Available Topics:")
    st.markdown(
        "<div class='sidebar-muted'>"
        "• Accident Reports<br>"
        "• Pothole Reports<br>"
        "• Accident Statistics"
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown("### Example Questions:")
    st.markdown(
        "<div class='sidebar-muted'>"
        "• \"Tell me about recent accidents\"<br>"
        "• \"What are the pothole reports?\"<br>"
        "• \"Show me accident statistics\""
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    # Health check 
    try:
        health_response = requests.get(HEALTH_URL, timeout=5)
        if health_response.status_code == 200:
            st.success("API Connected")
            st.json(health_response.json())
        else:
            st.error("API Error")
    except Exception:
        st.error("Cannot connect to API")

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    # Clear chat button 
    if st.button("Clear All History"):
        st.session_state.messages = []
        st.session_state.audio_messages = []
        st.rerun()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "audio_messages" not in st.session_state:
    st.session_state.audio_messages = []

# Display mode indicator 
st.markdown('<div class="rads-wrap">', unsafe_allow_html=True)
st.markdown(
    f'<div class="rads-card rads-pad"><span class="rads-pill">Current Mode: {chat_mode}</span></div>',
    unsafe_allow_html=True
)

#-----------------------------------------------
# >> TEXT CHAT MODE 
if chat_mode == "Text Chat":
    st.markdown('<div class="rads-wrap">', unsafe_allow_html=True)
    #st.markdown('<div class="rads-card rads-pad" style="margin-top:12px;">', unsafe_allow_html=True)
    st.markdown('<div class="rads-h2">Text Conversation</div>', unsafe_allow_html=True)

    # Display text chat history 
    st.markdown('<div class="rads-bubbles">', unsafe_allow_html=True)
    for message in st.session_state.messages:
        role = message["role"]
        cls = "user" if role == "user" else "assistant"
        st.markdown(f'<div class="rads-bubble {cls}">{message["content"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Text input 
    user_input = st.chat_input("Type your question here...")

    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.markdown(f'<div class="rads-bubble user">{user_input}</div>', unsafe_allow_html=True)

        # Get response
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    TEXT_API_URL,
                    json={"message": user_input},
                    timeout=120
                )

                if response.status_code == 200:
                    data = response.json()

                    if data["success"]:
                        assistant_message = data["response"]
                        st.markdown(f'<div class="rads-bubble assistant">{assistant_message}</div>', unsafe_allow_html=True)
                    else:
                        assistant_message = f" {data['response']}"
                        st.warning(assistant_message)
                else:
                    assistant_message = f"Error: {response.status_code}"
                    st.error(assistant_message)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_message
                })

            except requests.exceptions.Timeout:
                error_message = "Request timeout. Please try again."
                st.error(error_message)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message
                })

            except requests.exceptions.ConnectionError:
                error_message = "Cannot connect to API. Make sure backend is running."
                st.error(error_message)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message
                })

            except Exception as e:
                error_message = f"Error: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message
                })

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

#-----------------------------------------------
# >> AUDIO CHAT MODE 
elif chat_mode == "Voice Chat":
    st.markdown('<div class="rads-wrap">', unsafe_allow_html=True)
    #st.markdown('<div class="rads-card rads-pad" style="margin-top:12px;">', unsafe_allow_html=True)
    st.markdown('<div class="rads-h2">Voice Conversation</div>', unsafe_allow_html=True)

    # Audio recorder
    st.markdown("*Click the microphone to start/stop recording:*")
    audio_bytes = audio_recorder(
        text="Click to record",
        recording_color="#e74c3c",
        neutral_color="#3498db",
        icon_name="microphone",
        icon_size="3x",
    )

    # Process audio
    if audio_bytes:
        st.success("Audio recorded!")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Your Question:")
            st.audio(audio_bytes, format="audio/wav")

        # Send to backend
        with col2:
            with st.spinner("Processing your question..."):
                try:
                    files = {
                        "audio_file": ("recording.wav", io.BytesIO(audio_bytes), "audio/wav")
                    }

                    response = requests.post(
                        AUDIO_API_URL,
                        files=files,
                        timeout=120
                    )

                    if response.status_code == 200:
                        audio_response = response.content

                        if audio_response and len(audio_response) > 0:
                            st.success(f"Response received! ({len(audio_response)/1024:.1f} KB)")
                            st.markdown("#### Assistant's Answer:")

                            # Check WAV format
                            if audio_response[:4] == b'RIFF' and audio_response[8:12] == b'WAVE':
                                st.success("Valid WAV format")

                            # Play audio
                            try:
                                st.audio(audio_response, format="audio/wav", start_time=0)
                            except Exception as audio_error:
                                st.error(f"Playback error: {str(audio_error)}")

                                # Fallback: download button
                                st.download_button(
                                    label="Download Audio Response",
                                    data=audio_response,
                                    file_name="assistant_response.wav",
                                    mime="audio/wav"
                                )

                            # Add to history
                            st.session_state.audio_messages.append({
                                "user": audio_bytes,
                                "assistant": audio_response
                            })
                        else:
                            st.error("Empty audio response")

                    else:
                        st.error(f"Error: {response.status_code}")
                        try:
                            st.error(f"Details: {response.json().get('detail', 'Unknown')}")
                        except:
                            st.error(f"Response: {response.text[:500]}")

                except requests.exceptions.Timeout:
                    st.error("Request timeout. Please try again.")

                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API. Make sure backend is running on port 8000.")

                except Exception as e:
                    st.error(f"Error: {str(e)}")

    # Display audio history
    if len(st.session_state.audio_messages) > 0:
        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
        st.subheader("Voice Conversation History")

        for idx, message in enumerate(st.session_state.audio_messages):
            st.markdown(f"*Conversation {idx + 1}:*")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("Your Question:")
                st.audio(message["user"], format="audio/wav")

            with col2:
                st.markdown("Assistant's Answer:")
                st.audio(message["assistant"], format="audio/wav")

            st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="rads-wrap">', unsafe_allow_html=True)
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: var(--text-muted);'>
    <small>Unified Traffic Incident RAG System | Text & Voice | Powered by OpenAI & Agno</small>
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)