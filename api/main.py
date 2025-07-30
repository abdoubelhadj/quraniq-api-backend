import os
import logging
import sys
from contextlib import asynccontextmanager
import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Load environment variables
load_dotenv()

# Import chatbot
try:
    from app.chatbot import QuranIQChatbot
    chatbot_imported = True
    logging.info("✅ Successfully imported QuranIQChatbot from app.chatbot")
except ImportError as e:
    logging.error(f"❌ Failed to import QuranIQChatbot: {e}")
    chatbot_imported = False
    raise ImportError("Could not import QuranIQChatbot")

# Global chatbot instance
chatbot = None

# Rate limiting for the entire API
request_times = []
MAX_REQUESTS_PER_MINUTE = 10

def check_rate_limit():
    """Check if we're within rate limits"""
    current_time = time.time()
    global request_times
    request_times = [t for t in request_times if current_time - t < 60]
    if len(request_times) >= MAX_REQUESTS_PER_MINUTE:
        return False
    request_times.append(current_time)
    return True

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    global chatbot
    logging.info("🚀 Starting QuranIQ API...")
    logging.info(f"Current working directory: {os.getcwd()}")
    
    if chatbot_imported:
        try:
            chatbot = QuranIQChatbot()
            if hasattr(chatbot, 'is_loaded') and not chatbot.is_loaded:
                logging.warning("⚠️ Chatbot loaded but not initialized properly")
            else:
                logging.info("✅ QuranIQ API started successfully")
        except Exception as e:
            logging.error(f"❌ Failed to initialize QuranIQChatbot: {e}", exc_info=True)
            chatbot = None
    else:
        logging.error("❌ Cannot start chatbot - import failed")
    
    yield
    
    logging.info("🔄 Shutting down QuranIQ API...")
    if chatbot:
        chatbot = None
    logging.info("✅ QuranIQ API shutdown complete")

# Initialize FastAPI with lifespan
app = FastAPI(
    title="QuranIQ API",
    description="Chatbot Islamique avec Gemini",
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
    rate_limit_info: dict = None

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for monitoring"""
    rate_limit_info = {
        "requests_in_last_minute": len([t for t in request_times if time.time() - t < 60]),
        "max_requests_per_minute": MAX_REQUESTS_PER_MINUTE
    }
    
    if chatbot and hasattr(chatbot, 'is_loaded') and chatbot.is_loaded:
        return HealthResponse(
            status="healthy",
            message="QuranIQ API is running",
            model=getattr(chatbot, 'working_model_name', 'unknown'),
            rate_limit_info=rate_limit_info
        )
    elif chatbot_imported:
        return HealthResponse(
            status="degraded",
            message="QuranIQ API is running but chatbot not fully initialized",
            model="unknown",
            rate_limit_info=rate_limit_info
        )
    else:
        raise HTTPException(
            status_code=503, 
            detail="QuranIQ API is not fully initialized - import failed"
        )

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint"""
    return await health_check()

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(req: QueryRequest, request: Request):
    """Chat endpoint with rate limiting"""
    if not check_rate_limit():
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please wait before making another request."
        )
    
    if not chatbot:
        logging.error("Chatbot not initialized for request")
        raise HTTPException(
            status_code=500, 
            detail="Chatbot not initialized. Please check server logs."
        )
    
    try:
        logging.info(f"Processing chat request from {request.client.host}: {req.query[:50]}...")
        result = chatbot.chat(req.query)
        logging.info("Chat request processed successfully")
        return result
    except Exception as e:
        logging.error(f"Error processing chat request: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your request. Please try again later."
        )

@app.get("/ping")
async def ping():
    """Simple ping endpoint to keep the service alive"""
    return {"message": "pong", "timestamp": time.time()}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    logging.info(f"Starting server on port {port}")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True,
        keep_alive=True,
        timeout_keep_alive=30,
        timeout_graceful_shutdown=30,
        workers=1,
        reload=False
    )
