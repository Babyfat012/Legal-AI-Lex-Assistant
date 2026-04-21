import os
import uuid
from datetime import datetime
from sqlalchemy import Column, String, Boolean, DateTime, Text, ForeignKey, inspect
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import relationship

# Build DATABASE_URL, automatically switching to asyncpg if a sync URL is provided
raw_url = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/lex_db")
if raw_url.startswith("postgresql://"):
    DATABASE_URL = raw_url.replace("postgresql://", "postgresql+asyncpg://", 1)
else:
    DATABASE_URL = raw_url

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False)
    hashed_pw = Column(Text, nullable=False)
    full_name = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True) 
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))
    title = Column(String(500), default="Hội thoại mới")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Message(Base):
    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"))
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    sources = Column(JSONB, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class GeneratedDocument(Base):
    __tablename__ = "generated_documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))
    template_id = Column(String(100), nullable=False)
    template_name = Column(String(255), nullable=False)
    field_values = Column(JSONB, nullable=False)
    filename = Column(String(500), nullable=False)
    file_content = Column(Text, nullable=False)  # Base64 encoded content
    mime_type = Column(String(100), default="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    user = relationship("User")

async def init_auth_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
