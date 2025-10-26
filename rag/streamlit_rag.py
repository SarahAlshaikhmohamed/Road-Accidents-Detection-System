import streamlit as st
import requests
import json

# Page configuration
st.set_page_config(
    page_title="Traffic Incident RAG Chat",
    page_icon="🚗",
    layout="wide"
)

# API endpoint configuration
API_URL = "http://localhost:8000/chat"
HEALTH_URL = "http://localhost:8000/health"

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stTextInput > div > div > input {
        background-color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# App title and description
st.title("🚗 Traffic Incident RAG Assistant")
st.markdown("""
Welcome to the Traffic Incident Information System. Ask questions about:
- **Accident Reports** - Specific accident incidents and details
- **Pothole Reports** - Road conditions and pothole information
- **Accident Statistics** - Statistical data and trends
""")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ Information")
    st.markdown("""
    ### Available Topics:
    - Accident Reports
    - Pothole Reports
    - Accident Statistics
    
    ### How to Use:
    1. Type your question in the chat
    2. The system will find relevant information
    3. You'll get an answer from the knowledge base
    
    ### Example Questions:
    - "Tell me about recent accident reports"
    - "What are the pothole reports?"
    - "Show me accident statistics"
    """)
    
    # Check API health
    try:
        health_response = requests.get(HEALTH_URL, timeout=20)
        if health_response.status_code == 200:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Error")
    except:
        st.error("❌ Cannot connect to API")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Type your question here...")

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Display assistant response with loading indicator
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Send request to FastAPI backend
                response = requests.post(
                    API_URL,
                    json={"message": user_input},
                    timeout=120
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data["success"]:
                        # Successful response
                        assistant_message = data["response"]
                        st.markdown(assistant_message)
                    else:
                        # Guardrail violation or validation error
                        assistant_message = f"⚠️ {data['response']}"
                        st.warning(assistant_message)
                else:
                    # HTTP error
                    assistant_message = f"❌ Error: Unable to get response (Status: {response.status_code})"
                    st.error(assistant_message)
                
                # Add assistant message to chat history
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
                error_message = "❌ Cannot connect to API. Make sure the backend is running."
                st.error(error_message)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message
                })
                
            except Exception as e:
                error_message = f"❌ An error occurred: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message
                })

# Clear chat button in sidebar
with st.sidebar:
    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>Traffic Incident RAG System | Powered by OpenAI & Agno</small>
</div>
""", unsafe_allow_html=True)

# Run command: streamlit run app.py