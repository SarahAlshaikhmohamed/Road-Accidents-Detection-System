import streamlit as st
import requests
from audio_recorder_streamlit import audio_recorder
import io
import base64

# Page configuration
st.set_page_config(
    page_title="Traffic RAG Assistant",
    page_icon="🚗",
    layout="wide"
)

# API endpoint configuration
TEXT_API_URL = "http://localhost:8000/chat"
AUDIO_API_URL = "http://localhost:8000/audio-chat"
HEALTH_URL = "http://localhost:8000/health"

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .mode-selector {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# App title
st.title("🚗 Traffic Incident RAG Assistant")
st.markdown("""
**Unified System** - Ask questions using Text or Voice about:
- **Accident Reports** - Specific accident incidents
- **Pothole Reports** - Road conditions
- **Accident Statistics** - Statistical data and trends
""")

# Sidebar
with st.sidebar:
    st.header("ℹ️ Information")
    
    # Mode selector in sidebar
    st.subheader("🎛️ Select Mode")
    chat_mode = st.radio(
        "Choose interaction mode:",
        ["💬 Text Chat", "🎤 Voice Chat"],
        help="Switch between text typing and voice recording"
    )
    
    st.markdown("---")
    st.markdown("""
    ### Available Topics:
    - Accident Reports
    - Pothole Reports
    - Accident Statistics
    
    ### Example Questions:
    - "Tell me about recent accidents"
    - "What are the pothole reports?"
    - "Show me accident statistics"
    """)
    
    st.markdown("---")
    
    # Health check
    try:
        health_response = requests.get(HEALTH_URL, timeout=5)
        if health_response.status_code == 200:
            st.success("✅ API Connected")
            health_data = health_response.json()
            st.json(health_data)
        else:
            st.error("❌ API Error")
    except:
        st.error("❌ Cannot connect to API")
    
    st.markdown("---")
    
    # Clear chat button
    if st.button("🗑️ Clear All History"):
        st.session_state.messages = []
        st.session_state.audio_messages = []
        st.rerun()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "audio_messages" not in st.session_state:
    st.session_state.audio_messages = []

# Display mode indicator
st.markdown(f"<div class='mode-selector'><h3>Current Mode: {chat_mode}</h3></div>", unsafe_allow_html=True)

# ==================== TEXT CHAT MODE ====================
if chat_mode == "💬 Text Chat":
    st.markdown("### 💬 Text Conversation")
    
    # Display text chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Text input
    user_input = st.chat_input("Type your question here...")
    
    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(
                        TEXT_API_URL,
                        json={"message": user_input},
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        if data["success"]:
                            assistant_message = data["response"]
                            st.markdown(assistant_message)
                        else:
                            assistant_message = f"⚠️ {data['response']}"
                            st.warning(assistant_message)
                    else:
                        assistant_message = f"❌ Error: {response.status_code}"
                        st.error(assistant_message)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": assistant_message
                    })
                    
                except requests.exceptions.Timeout:
                    error_message = "❌ Request timeout. Please try again."
                    st.error(error_message)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message
                    })
                    
                except requests.exceptions.ConnectionError:
                    error_message = "❌ Cannot connect to API. Make sure backend is running."
                    st.error(error_message)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message
                    })
                    
                except Exception as e:
                    error_message = f"❌ Error: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message
                    })

# ==================== AUDIO CHAT MODE ====================
elif chat_mode == "🎤 Voice Chat":
    st.markdown("### 🎤 Voice Conversation")
    
    # Audio recorder
    st.markdown("**Click the microphone to start/stop recording:**")
    audio_bytes = audio_recorder(
        text="Click to record",
        recording_color="#e74c3c",
        neutral_color="#3498db",
        icon_name="microphone",
        icon_size="3x",
    )
    
    # Process audio
    if audio_bytes:
        st.success("✅ Audio recorded!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎙️ Your Question:")
            st.audio(audio_bytes, format="audio/wav")
        
        # Send to backend
        with col2:
            with st.spinner("🤔 Processing your question..."):
                try:
                    files = {
                        "audio_file": ("recording.wav", io.BytesIO(audio_bytes), "audio/wav")
                    }
                    
                    response = requests.post(
                        AUDIO_API_URL,
                        files=files,
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        audio_response = response.content
                        
                        if audio_response and len(audio_response) > 0:
                            st.success(f"✅ Response received! ({len(audio_response)/1024:.1f} KB)")
                            st.markdown("#### 🔊 Assistant's Answer:")
                            
                            # Check WAV format
                            if audio_response[:4] == b'RIFF' and audio_response[8:12] == b'WAVE':
                                st.success("✅ Valid WAV format")
                            
                            # Play audio
                            try:
                                st.audio(audio_response, format="audio/wav", start_time=0)
                            except Exception as audio_error:
                                st.error(f"Playback error: {str(audio_error)}")
                                
                                # Fallback: download button
                                st.download_button(
                                    label="⬇️ Download Audio Response",
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
                            st.error("❌ Empty audio response")
                    
                    else:
                        st.error(f"❌ Error: {response.status_code}")
                        try:
                            st.error(f"Details: {response.json().get('detail', 'Unknown')}")
                        except:
                            st.error(f"Response: {response.text[:500]}")
                
                except requests.exceptions.Timeout:
                    st.error("❌ Request timeout. Please try again.")
                
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to API. Make sure backend is running on port 8000.")
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # Display audio history
    if len(st.session_state.audio_messages) > 0:
        st.markdown("---")
        st.subheader("📝 Voice Conversation History")
        
        for idx, message in enumerate(st.session_state.audio_messages):
            st.markdown(f"**Conversation {idx + 1}:**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("*Your Question:*")
                st.audio(message["user"], format="audio/wav")
            
            with col2:
                st.markdown("*Assistant's Answer:*")
                st.audio(message["assistant"], format="audio/wav")
            
            st.markdown("---")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>Unified Traffic Incident RAG System | Text & Voice | Powered by OpenAI & Agno</small>
</div>
""", unsafe_allow_html=True)

# Run command: streamlit run unified_app.py