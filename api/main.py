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

# Import chatbot with multiple fallback strategies
chatbot_imported = False
QuranIQChatbot = None

# Strategy 1: Import from app.chatbot (correct path for your structure)
try:
    from app.chatbot import QuranIQChatbot
    chatbot_imported = True
    logging.info("✅ Successfully imported QuranIQChatbot from app.chatbot")
except ImportError as e:
    logging.warning(f"⚠️ Failed to import from app.chatbot: {e}")

# Strategy 2: Import from api.app.chatbot (alternative path)
if not chatbot_imported:
    try:
        from api.app.chatbot import QuranIQChatbot
        chatbot_imported = True
        logging.info("✅ Successfully imported QuranIQChatbot from api.app.chatbot")
    except ImportError as e:
        logging.warning(f"⚠️ Failed to import from api.app.chatbot: {e}")

# Strategy 3: Add current directory to path and try again
if not chatbot_imported:
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)
        from app.chatbot import QuranIQChatbot
        chatbot_imported = True
        logging.info("✅ Successfully imported QuranIQChatbot after adding current dir to path")
    except ImportError as e:
        logging.warning(f"⚠️ Failed to import after adding current dir: {e}")

# Strategy 4: Try direct file import
if not chatbot_imported:
    try:
        import importlib.util
        chatbot_path = os.path.join(os.path.dirname(__file__), 'app', 'chatbot.py')
        if os.path.exists(chatbot_path):
            spec = importlib.util.spec_from_file_location("chatbot", chatbot_path)
            chatbot_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(chatbot_module)
            QuranIQChatbot = chatbot_module.QuranIQChatbot
            chatbot_imported = True
            logging.info("✅ Successfully imported QuranIQChatbot via direct file import")
    except Exception as e:
        logging.warning(f"⚠️ Failed direct file import: {e}")

# Final check
if not chatbot_imported or QuranIQChatbot is None:
    logging.error("❌ All import strategies failed!")
    # Create a dummy class to prevent startup failure
    class QuranIQChatbot:
        def __init__(self):
            self.is_loaded = False
        def chat(self, query):
            return {"response": "Service temporarily unavailable", "language": "fr", "sources": [], "mode": "error"}

# Global chatbot instance
chatbot = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    global chatbot
    
    # Startup
    logging.info("🚀 Starting QuranIQ API...")
    logging.info(f"Current working directory: {os.getcwd()}")
    logging.info(f"Python path: {sys.path[:3]}")
    logging.info(f"Chatbot imported successfully: {chatbot_imported}")
    
    # List files in current directory for debugging
    try:
        current_files = os.listdir('.')
        logging.info(f"Files in current directory: {current_files}")
        if 'app' in current_files:
            app_files = os.listdir('./app')
            logging.info(f"Files in app directory: {app_files}")
    except Exception as e:
        logging.warning(f"Could not list directory contents: {e}")
    
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

class DebugResponse(BaseModel):
    current_directory: str
    python_path: list
    files_structure: dict
    chatbot_status: dict
    environment_vars: dict

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for monitoring"""
    if chatbot and hasattr(chatbot, 'is_loaded') and chatbot.is_loaded:
        return HealthResponse(
            status="healthy",
            message="QuranIQ API is running",
            model=getattr(chatbot, 'working_model_name', 'unknown')
        )
    elif chatbot_imported:
        return HealthResponse(
            status="degraded",
            message="QuranIQ API is running but chatbot not fully initialized",
            model="unknown"
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
async def chat_endpoint(req: QueryRequest):
    """Chat endpoint"""
    if not chatbot:
        logging.error("Chatbot not initialized for request")
        raise HTTPException(
            status_code=500, 
            detail="Chatbot not initialized. Please check server logs."
        )
    
    if not chatbot_imported:
        raise HTTPException(
            status_code=500,
            detail="Chatbot module could not be imported. Please check deployment."
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

@app.get("/debug/info", response_model=DebugResponse)
async def debug_info():
    """Debug endpoint to check system information"""
    import os
    import sys
    
    current_dir = os.getcwd()
    
    # Build file structure
    files_structure = {}
    try:
        for root, dirs, files in os.walk(current_dir):
            rel_root = os.path.relpath(root, current_dir)
            if rel_root == '.':
                rel_root = 'root'
            files_structure[rel_root] = {
                'directories': dirs,
                'files': [f for f in files if f.endswith(('.py', '.txt', '.yaml', '.yml', '.json'))]
            }
            # Limit depth to avoid too much data
            if len(files_structure) > 10:
                break
    except Exception as e:
        files_structure = {"error": str(e)}
    
    return DebugResponse(
        current_directory=current_dir,
        python_path=sys.path[:5],
        files_structure=files_structure,
        chatbot_status={
            "imported": chatbot_imported,
            "instance_created": chatbot is not None,
            "initialized": getattr(chatbot, 'is_loaded', False) if chatbot else False,
            "model": getattr(chatbot, 'working_model_name', 'N/A') if chatbot else 'N/A'
        },
        environment_vars={
            "PORT": os.getenv("PORT", "Not set"),
            "GOOGLE_API_KEY_SET": "Yes" if os.getenv("GOOGLE_GENERATIVE_AI_API_KEY") else "No",
            "COHERE_API_KEY_SET": "Yes" if os.getenv("COHERE_API_KEY") else "No",
            "BLOB_URLS_SET": "Yes" if (os.getenv("BLOB_INDEX_URL") and os.getenv("BLOB_METADATA_URL")) else "No"
        }
    )

# Graceful shutdown handler
def signal_handler(signum, frame):
    logging.info(f"Received signal {signum}, shutting down gracefully...")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment (Render sets PORT automatically)
    port = int(os.getenv("PORT", 10000))
    
    logging.info(f"Starting server on port {port}")
    
    # Run with proper configuration for production
    uvicorn.run(
        "main:app",  # Changed from "api.main:app" to "main:app"
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
