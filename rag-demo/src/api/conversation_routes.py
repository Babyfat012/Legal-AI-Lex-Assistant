"""
Conversation API Routes — CRUD for chat conversations.

Replaces Chainlit's data_layer.py with REST endpoints.
Messages are stored in PostgreSQL (same tables: conversations, messages).
"""

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select, delete, func
from sqlalchemy.dialects.postgresql import insert

from auth.database import AsyncSessionLocal, Conversation, Message, User
from api.auth_routes import get_current_user
from core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Conversations"])


# --- Schemas ---

class ConversationOut(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str


class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    sources: list | None = None
    created_at: str


class ConversationDetail(BaseModel):
    conversation: ConversationOut
    messages: list[MessageOut]


class CreateConversationRequest(BaseModel):
    title: str = Field(default="Hội thoại mới")


# --- Endpoints ---

@router.get("/conversations", response_model=list[ConversationOut])
async def list_conversations(current_user: dict = Depends(get_current_user)):
    """List all conversations for the current user (sidebar)."""
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Conversation)
            .where(Conversation.user_id == uuid.UUID(current_user["user_id"]))
            .order_by(Conversation.updated_at.desc())
            .limit(50)
        )
        conversations = result.scalars().all()

    return [
        ConversationOut(
            id=str(c.id),
            title=c.title or "Hội thoại mới",
            created_at=c.created_at.isoformat() if c.created_at else "",
            updated_at=c.updated_at.isoformat() if c.updated_at else "",
        )
        for c in conversations
    ]


@router.post("/conversations", response_model=ConversationOut)
async def create_conversation(
    req: CreateConversationRequest = CreateConversationRequest(),
    current_user: dict = Depends(get_current_user),
):
    """Create a new conversation."""
    conv_id = uuid.uuid4()
    now = datetime.utcnow()

    async with AsyncSessionLocal() as db:
        conv = Conversation(
            id=conv_id,
            user_id=uuid.UUID(current_user["user_id"]),
            title=req.title,
            created_at=now,
            updated_at=now,
        )
        db.add(conv)
        await db.commit()

    logger.info("Created conversation | id=%s | user=%s", conv_id, current_user["email"])
    return ConversationOut(
        id=str(conv_id),
        title=req.title,
        created_at=now.isoformat(),
        updated_at=now.isoformat(),
    )


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Get a conversation with all its messages (resume chat)."""
    async with AsyncSessionLocal() as db:
        # Verify ownership
        result = await db.execute(
            select(Conversation).where(
                Conversation.id == uuid.UUID(conversation_id),
                Conversation.user_id == uuid.UUID(current_user["user_id"]),
            )
        )
        conv = result.scalar_one_or_none()
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Load messages
        msg_result = await db.execute(
            select(Message)
            .where(Message.conversation_id == conv.id)
            .order_by(Message.created_at.asc())
        )
        messages = msg_result.scalars().all()

    return ConversationDetail(
        conversation=ConversationOut(
            id=str(conv.id),
            title=conv.title or "Hội thoại mới",
            created_at=conv.created_at.isoformat() if conv.created_at else "",
            updated_at=conv.updated_at.isoformat() if conv.updated_at else "",
        ),
        messages=[
            MessageOut(
                id=str(m.id),
                role=m.role,
                content=m.content,
                sources=m.sources,
                created_at=m.created_at.isoformat() if m.created_at else "",
            )
            for m in messages
        ],
    )


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Delete a conversation and all its messages."""
    async with AsyncSessionLocal() as db:
        # Verify ownership
        result = await db.execute(
            select(Conversation).where(
                Conversation.id == uuid.UUID(conversation_id),
                Conversation.user_id == uuid.UUID(current_user["user_id"]),
            )
        )
        conv = result.scalar_one_or_none()
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")

        await db.execute(
            delete(Message).where(Message.conversation_id == conv.id)
        )
        await db.execute(
            delete(Conversation).where(Conversation.id == conv.id)
        )
        await db.commit()

    logger.info("Deleted conversation | id=%s", conversation_id)
    return {"status": "deleted"}
