import os
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import UUID
import uuid

# Get database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/lex_db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class IngestLog(Base):
    __tablename__ = "ingest_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    file_name = Column(String(500), nullable=False)
    upload_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String(20), nullable=False)   # 'success' | 'failed' | 'deleted'
    chunk_count = Column(Integer, default=0)
    elapsed_secs = Column(Integer, default=0)
    error_msg = Column(Text, nullable=True)
    uploaded_by = Column(String(255), default='admin')

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
