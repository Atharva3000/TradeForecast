from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging

from services.karen import generate_karen_response

logger = logging.getLogger(__name__)
router = APIRouter()

class ChatMessage(BaseModel):
    role: str # 'user' or 'assistant'
    content: str

class ChatRequest(BaseModel):
    username: str
    message: str
    history: list[ChatMessage] = []
    active_ticker: Optional[str] = None
    api_key: Optional[str] = None

@router.post("/api/chat/karen")
async def chat_with_karen(req: ChatRequest):
    """
    POST route to interact with Karen AI assistant.
    Returns assistant message and platform trigger hooks.
    """
    try:
        # Convert Pydantic schemas to standard dictionaries
        history_list = [{"role": h.role, "content": h.content} for h in req.history]
        
        response = await generate_karen_response(
            username=req.username,
            message=req.message,
            history=history_list,
            active_ticker=req.active_ticker,
            user_api_key=req.api_key
        )
        return response
    except Exception as e:
        logger.error("Error in /api/chat/karen endpoint: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
