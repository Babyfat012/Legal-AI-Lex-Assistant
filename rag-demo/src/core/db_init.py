import os
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine
from core.database import Base, engine as main_engine
from auth.database import Base as AuthBase, engine as auth_engine

def init_all_databases():
    """
    Initialize both main and authentication databases
    """
    # Initialize main database (sync)
    Base.metadata.create_all(bind=main_engine)

    # Initialize auth database (async) - run in event loop
    import asyncio
    try:
        # If we're already in an event loop
        asyncio.get_event_loop().run_until_complete(_init_auth_db())
    except RuntimeError:
        # If no event loop exists, create one
        asyncio.run(_init_auth_db())

async def _init_auth_db():
    """
    Initialize only the auth database (async)
    """
    async with auth_engine.begin() as conn:
        await conn.run_sync(AuthBase.metadata.create_all)

def init_main_db():
    """
    Initialize only the main database (for compatibility)
    """
    Base.metadata.create_all(bind=main_engine)