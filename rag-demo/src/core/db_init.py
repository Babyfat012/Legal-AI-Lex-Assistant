import os
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine
from core.database import Base, engine as main_engine
from auth.database import Base as AuthBase, engine as auth_engine

async def init_all_databases():
    """
    Initialize both main and authentication databases
    """
    # Initialize main database (sync)
    Base.metadata.create_all(bind=main_engine)

    # Initialize auth database (async)
    async with auth_engine.begin() as conn:
        await conn.run_sync(AuthBase.metadata.create_all)

def init_main_db():
    """
    Initialize only the main database (for compatibility)
    """
    Base.metadata.create_all(bind=main_engine)