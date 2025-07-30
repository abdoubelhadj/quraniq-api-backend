import os
import logging
import sys
from contextlib import asynccontextmanager
import signal
import asyncio

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import chatbot after environment setup
try:
    from app.chatbot import QuranIQChatbot
except ImportError:
    from chatbot import QuranIQChatbot

# Global chatbot instance
chatbot = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    global chatbot
    
    # Startup
    logging.info("🚀 Starting QuranIQ API...")
    try:
        chatbot = QuranIQChatbot()
        if not chatbot.is_loaded:
            raise Exception("Chatbot failed to load during startup.")
        logging.info("✅ QuranIQ API started successfully")
    except Exception as e:
        logging.error(f"❌ Failed to initialize QuranIQChatbot: {e}", exc_info=True)
        chatbot = None
    
    yield
    
    # Shutdown
    logging.info("🔄 Shutting down QuranIQ API...")
    if chatbot:
        chatbot = None
    logging.info("✅ QuranIQ API shutdown complete")

# Initialize FastAPI with lifespan
app = FastAPI(
    title="QuranIQ API",
    description="Chatbot Islamique avec Gemini et RAG",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Pydantic schemas
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str
    language: str
    sources: list
    mode: str

class HealthResponse(BaseModel):
    status: str
    message: str
    model: str = None

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for monitoring"""
    if chatbot and chatbot.is_loaded:
        return HealthResponse(
            status="healthy",
            message="QuranIQ API is running",
            model=getattr(chatbot, 'working_model_name', 'unknown')
        )
    else:
        raise HTTPException(
            status_code=503, 
            detail="QuranIQ API is not fully initialized"
        )

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint"""
    return await health_check()

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(req: QueryRequest):
    """Chat endpoint"""
    if not chatbot or not chatbot.is_loaded:
        logging.error("Chatbot not initialized for request")
        raise HTTPException(
            status_code=500, 
            detail="Chatbot not initialized. Please check server logs."
        )
    
    try:
        logging.info(f"Processing chat request: {req.query[:50]}...")
        result = chatbot.chat(req.query)
        logging.info("Chat request processed successfully")
        return result
    except Exception as e:
        logging.error(f"Error processing chat request: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your request"
        )

@app.get("/ping")
async def ping():
    """Simple ping endpoint to keep the service alive"""
    return {"message": "pong", "timestamp": asyncio.get_event_loop().time()}

# Graceful shutdown handler
def signal_handler(signum, frame):
    logging.info(f"Received signal {signum}, shutting down gracefully...")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Alternative main.py at root level for flexibility
if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment (Render sets PORT automatically)
    port = int(os.getenv("PORT", 10000))
    
    logging.info(f"Starting server on port {port}")
    
    # Run with proper configuration for production
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
