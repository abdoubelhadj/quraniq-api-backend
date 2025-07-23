import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv # Import load_dotenv
from .app.chatbot import QuranIQChatbot # Correct import path for the recommended structure

# Load environment variables from .env file (for local development)
load_dotenv()

# Init app
app = FastAPI(
    title="QuranIQ API",
    description="Chatbot Islamique avec Gemini + RAG",
    version="1.0.0"
)

# CORS configuration
# For production, replace "*" with specific origins (e.g., your Android app's domain)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Be more restrictive in production
    allow_methods=["*"],
    allow_headers=["*"]
)

# Chatbot global instance
# This will load models only once when the API starts
try:
    chatbot = QuranIQChatbot()
    if not chatbot.is_loaded:
        raise Exception("Chatbot failed to load during startup.")
except Exception as e:
    # Log the error and set chatbot to None to indicate failure
    import logging
    logging.error(f"Failed to initialize QuranIQChatbot at startup: {e}", exc_info=True)
    chatbot = None # Indicate that the chatbot is not ready

# Pydantic schemas for request and response
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str
    language: str
    sources: list
    mode: str

# Endpoints
@app.get("/")
async def root():
    if chatbot and chatbot.is_loaded:
        return {"message": "QuranIQ API is running", "status": "ok", "model": chatbot.working_model_name}
    else:
        raise HTTPException(status_code=503, detail="QuranIQ API is not fully initialized or encountered an error during startup.")

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(req: QueryRequest):
    if not chatbot or not chatbot.is_loaded:
        raise HTTPException(status_code=500, detail="Chatbot not initialized. Please check server logs for errors.")
    
    # Call the chat method from the chatbot instance
    return chatbot.chat(req.query)
