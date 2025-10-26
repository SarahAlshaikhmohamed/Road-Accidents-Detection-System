from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
import asyncio

# FastAPI app initialization
app = FastAPI(title="Traffic Incident RAG API")

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database and API configuration
db_url = "postgresql://postgres:RuXVDgIYQ0X0ddPf@db.ewbtzrfwoteqyvzkyzhk.supabase.co:5432/postgres"
api_key_openai = "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"

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

# Create knowledge bases for three different topics
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

# Custom retriever that routes queries to the correct knowledge base
async def topic_router(agent: Agent, query: str, num_documents: int = 5, **kwargs):
    """Route the query to the appropriate knowledge base based on keywords."""
    if "accident_reports" in query.lower() or "accident reports" in query.lower():
        docs = await knowledge_Accident_reports.async_search(query, max_results=num_documents)
    elif "pothole_report" in query.lower() or "pothole report" in query.lower():
        docs = await knowledge_Pothole.async_search(query, max_results=num_documents)
    elif "accident_statistics" in query.lower() or "accident statistics" in query.lower():
        docs = await knowledge_Accident_Statistics.async_search(query, max_results=num_documents)
    else:
        # Default search across all knowledge bases
        docs = await knowledge_Accident_Statistics.async_search(query, max_results=num_documents)
    
    return [d.to_dict() for d in docs]

# Initialize database connection
db = PostgresDb(db_url=db_url)

# Create RAG Agent
RAG_agent = Agent(
    name="RAG agent",
    model=OpenAIChat(id="gpt-4o", api_key=api_key_openai, modalities=["text"]),
    knowledge_retriever=topic_router,
    search_knowledge=True,
    pre_hooks=[PromptInjectionGuardrail(), TrafficIncidentGuardrail()],
    db=db,
    read_chat_history=True,
    add_history_to_context=True,
    num_history_runs=7,
    instructions="""
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
5. Do NOT generate or guess any information that is not supported by the retrieved local data.
6. Do not add your own information.
""",
)

# API Endpoints
@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Traffic Incident RAG API is running", "status": "active"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint that processes user queries through the RAG agent.
    
    Args:
        request: ChatRequest containing user message
        
    Returns:
        ChatResponse with agent's response and success status
    """
    try:
        # Run the RAG agent with user's message
        response = await RAG_agent.arun(request.message)
        
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

@app.get("/health")
async def health_check():
    """Check if all services are working."""
    return {
        "api": "healthy",
        "database": "connected",
        "agent": "ready"
    }

# Run the FastAPI server
# Command: uvicorn backend:app --reload --host 0.0.0.0 --port 8000