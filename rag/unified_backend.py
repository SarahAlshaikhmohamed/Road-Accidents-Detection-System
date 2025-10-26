from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from agno.agent import Agent
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.pgvector import PgVector, SearchType
from agno.db.postgres import PostgresDb
from agno.models.openai import OpenAIChat
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.exceptions import CheckTrigger, InputCheckError
from agno.guardrails import BaseGuardrail, PromptInjectionGuardrail
from agno.run.team import TeamRunInput
from agno.media import Audio
import base64
import os
from dotenv import load_dotenv
load_dotenv()

# FastAPI app initialization
app = FastAPI(title="Unified Traffic Incident RAG API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database and API configuration
db_url = os.getenv("DATABASE_URL")
api_key_openai  = os.getenv("OPENAI_API_KEY")

# Request and Response models
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    success: bool

# Guardrail for topic relevance
class TopicRelevanceResult(BaseModel):
    is_relevant: bool
    reason: str

class TrafficIncidentGuardrail(BaseGuardrail):
    """Guardrail to check if query is related to traffic incidents."""
    
    def check(self, run_input: TeamRunInput) -> None:
        if isinstance(run_input.input_content, str):
            validator = Agent(
                model=OpenAIChat(id="gpt-4o-mini", api_key=api_key_openai),
                instructions=[
                    "Determine if the user's question is related to traffic incidents, accidents, road safety, or transportation data.",
                    "Consider synonyms like: mobility, collisions, crashes, road events, transportation incidents.",
                    "Return is_relevant=True if related, False otherwise."
                ],
                output_schema=TopicRelevanceResult,
            )
            
            result = validator.run(
                input=f"Is this question about traffic incidents? '{run_input.input_content}'"
            ).content
            
            if not result.is_relevant:
                raise InputCheckError(
                    f"This system only handles traffic incident queries. {result.reason}",
                    check_trigger=CheckTrigger.OFF_TOPIC,
                )
    
    async def async_check(self, run_input: TeamRunInput) -> None:
        self.check(run_input)

# Initialize embeddings
embeddings = OpenAIEmbedder()

# Create knowledge bases
knowledge_Accident_Statistics = Knowledge(
    vector_db=PgVector(
        table_name="Traffic_Accident_Statistics_openai",
        db_url=db_url,
        embedder=embeddings,
        search_type=SearchType.hybrid
    ),
    max_results=5
)

knowledge_Pothole = Knowledge(
    vector_db=PgVector(
        table_name="Pothole_report",
        db_url=db_url,
        embedder=embeddings,
        search_type=SearchType.hybrid
    ),
    max_results=5
)

knowledge_Accident_reports = Knowledge(
    vector_db=PgVector(
        table_name="Accident_reports",
        db_url=db_url,
        embedder=embeddings,
        search_type=SearchType.hybrid
    ),
    max_results=5
)

# Topic router for knowledge base selection
async def topic_router(agent: Agent, query: str, num_documents: int = 5, **kwargs):
    """Route the query to the appropriate knowledge base."""
    if "accident_reports" in query.lower() or "accident reports" in query.lower():
        docs = await knowledge_Accident_reports.async_search(query, max_results=num_documents)
    elif "pothole_report" in query.lower() or "pothole report" in query.lower():
        docs = await knowledge_Pothole.async_search(query, max_results=num_documents)
    elif "accident_statistics" in query.lower() or "accident statistics" in query.lower():
        docs = await knowledge_Accident_Statistics.async_search(query, max_results=num_documents)
    else:
        docs = await knowledge_Accident_reports.async_search(query, max_results=num_documents)
    
    return [d.to_dict() for d in docs]

# Initialize database
db = PostgresDb(db_url=db_url)

# Shared instructions for both agents
AGENT_INSTRUCTIONS = """
You are a RAG Agent specialized in answering questions strictly based on a **local knowledge base** stored in a Vector Database.
There are three available topics, and you must decide which one the user's query belongs to before answering:

- Accident_reports → For questions about specific car accidents or accident reports (e.g., time, location, details of an incident).
- Pothole_report → For questions about road conditions, potholes, or street surface issues.
- Accident_Statistics → For questions about accident numbers, yearly or monthly statistics, or overall accident trends.

Your rules:
1. Always use information ONLY from the local knowledge base (the vector DB). Never use outside or general knowledge.
2. Always identify the correct topic and include it at the start of your answer as: "SelectedTopic: <topic_name>".
3. If relevant information is found, answer clearly and concisely using the retrieved content as evidence.
4. If no relevant information is found in the local knowledge base, respond with an apology such as:
   "Sorry, I could not find any matching information in the local knowledge base."
5. DO NOT generate or guess any information that is not supported by the retrieved local data.
6. Do not add your own information.

When responding via audio, speak clearly and naturally.
"""

# Create TEXT Agent (for text chat)
text_agent = Agent(
    name="Text RAG Agent",
    model=OpenAIChat(id="gpt-4o", api_key=api_key_openai, modalities=["text"]),
    knowledge_retriever=topic_router,
    search_knowledge=True,
    pre_hooks=[PromptInjectionGuardrail(), TrafficIncidentGuardrail()],
    db=db,
    read_chat_history=True,
    add_history_to_context=True,
    num_history_runs=7,
    instructions=AGENT_INSTRUCTIONS,
)

# Create AUDIO Agent (for voice chat)
audio_agent = Agent(
    name="Audio RAG Agent",
    model=OpenAIChat(
        id="gpt-4o-audio-preview",
        modalities=["text", "audio"],
        audio={"voice": "sage", "format": "wav"},
        api_key=api_key_openai
    ),
    knowledge_retriever=topic_router,
    search_knowledge=True,
    #pre_hooks=[PromptInjectionGuardrail(), TrafficIncidentGuardrail()],
    db=db,
    read_chat_history=True,
    add_history_to_context=True,
    num_history_runs=7,
    instructions=AGENT_INSTRUCTIONS,
)

# API Endpoints
@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "Unified Traffic Incident RAG API is running",
        "status": "active",
        "endpoints": {
            "text_chat": "/chat",
            "audio_chat": "/audio-chat",
            "health": "/health"
        }
    }

@app.post("/chat", response_model=ChatResponse)
async def text_chat(request: ChatRequest):
    """
    TEXT chat endpoint - processes text queries and returns text responses.
    
    Args:
        request: ChatRequest containing user message
        
    Returns:
        ChatResponse with agent's text response
    """
    try:
        # Run the TEXT agent with user's message
        response = await text_agent.arun(request.message)
        
        # Extract response content
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        return ChatResponse(
            response=response_text,
            success=True
        )
    
    except InputCheckError as e:
        # Handle guardrail violations
        return ChatResponse(
            response=str(e),
            success=False
        )
    
    except Exception as e:
        # Handle other errors
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/audio-chat")
async def voice_chat(audio_file: UploadFile = File(...)):
    """
    AUDIO chat endpoint - processes audio input and returns audio response.
    
    Args:
        audio_file: Audio file from user (WAV format)
        
    Returns:
        Audio response in WAV format
    """
    try:
        # Read audio file content
        audio_content = await audio_file.read()
        
        # Run AUDIO agent with audio input
        response = await audio_agent.arun(
            "Please answer the question in this audio recording based on the knowledge base.",
            audio=[Audio(content=audio_content, format="wav")]
        )
        
        # Manual parsing of response for audio content
        audio_response_content = None
        
        try:
            response_dict = response.__dict__
            if 'response_audio' in response_dict:
                resp_audio = response_dict['response_audio']
                if resp_audio and hasattr(resp_audio, 'content'):
                    audio_response_content = resp_audio.content
                elif isinstance(resp_audio, dict) and 'content' in resp_audio:
                    audio_response_content = resp_audio['content']
            
            # Check 'audio' field (alternative location)
            if not audio_response_content and 'audio' in response_dict:
                audio_list = response_dict['audio']
                if audio_list and len(audio_list) > 0:
                    if hasattr(audio_list[0], 'content'):
                        audio_response_content = audio_list[0].content
        
        except Exception as parse_error:
            raise HTTPException(
                status_code=500,
                detail=f"Error parsing audio response: {str(parse_error)}"
            )
        
        if not audio_response_content:
            raise HTTPException(
                status_code=500,
                detail="No audio content found in response"
            )
        
        # Decode base64 if needed
        final_audio_bytes = None
        
        try:
            if isinstance(audio_response_content, str):
                print("🔄 Decoding base64 audio...")
                final_audio_bytes = base64.b64decode(audio_response_content)
                print(f"✅ Decoded {len(final_audio_bytes)} bytes from base64")
            
            elif isinstance(audio_response_content, bytes):
                if not audio_response_content.startswith(b'RIFF'):
                    print("🔄 Attempting to decode base64 from bytes...")
                    try:
                        final_audio_bytes = base64.b64decode(audio_response_content)
                        print(f"✅ Decoded {len(final_audio_bytes)} bytes from base64")
                    except:
                        print("⚠️ Not base64, using raw bytes")
                        final_audio_bytes = audio_response_content
                else:
                    print("✅ Already valid WAV bytes")
                    final_audio_bytes = audio_response_content
        
        except Exception as decode_error:
            print(f"⚠️ Decode error: {decode_error}")
            final_audio_bytes = audio_response_content
        
        if not final_audio_bytes or len(final_audio_bytes) == 0:
            raise HTTPException(
                status_code=500,
                detail="Audio decoding resulted in empty content"
            )
        
        print(f"✅ Final audio size: {len(final_audio_bytes)} bytes")
        
        # Fix WAV header if needed
        fixed_audio = final_audio_bytes
        try:
            if final_audio_bytes[:4] == b'RIFF':
                actual_size = len(final_audio_bytes) - 8
                fixed_audio = b'RIFF' + actual_size.to_bytes(4, 'little') + final_audio_bytes[8:]
                print(f"✅ Fixed WAV header")
        except Exception as fix_error:
            print(f"⚠️ Header fix error: {fix_error}")
        
        return Response(
            content=fixed_audio,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "inline; filename=response.wav",
                "Content-Length": str(len(fixed_audio)),
                "Accept-Ranges": "bytes",
                "Cache-Control": "no-cache"
            }
        )
    
    except InputCheckError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

@app.get("/health")
async def health_check():
    """Check if all services are working."""
    return {
        "api": "healthy",
        "database": "connected",
        "text_agent": "ready",
        "audio_agent": "ready"
    }

# Run command: uvicorn unified_backend:app --reload --host 0.0.0.0 --port 8000