"""
Auth API Routes — JWT-based authentication.

Reuses existing User table from auth/database.py + security.py.
"""

import os
import uuid
from datetime import datetime, timedelta
import re

import jwt
from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel, Field, validator
from sqlalchemy import select

from auth.database import AsyncSessionLocal, User
from auth.security import verify_password, hash_password
from core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Auth"])

JWT_SECRET = os.getenv("JWT_SECRET_KEY", "change-me-in-production")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))


# --- Schemas ---

class LoginRequest(BaseModel):
    email: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class RegisterRequest(BaseModel):
    email: str = Field(..., min_length=1)
    password: str = Field(..., min_length=6)
    full_name: str = Field(default="")

    @validator('email')
    def validate_email(cls, v):
        if not re.match(r'^[^@]+@[^@]+\.[^@]+$', v):
            raise ValueError('Invalid email format')
        return v.lower()


class AuthResponse(BaseModel):
    access_token: str
    user: dict


class UserInfo(BaseModel):
    id: str
    email: str
    full_name: str


# --- Helpers ---

def _create_token(user_id: str, email: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def _decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


async def get_current_user(authorization: str = Header(None)) -> dict:
    """Dependency: extract user from JWT token in Authorization header."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = authorization.split(" ", 1)[1]
    payload = _decode_token(token)
    return {"user_id": payload["sub"], "email": payload["email"]}


# --- Endpoints ---

@router.post("/login", response_model=AuthResponse)
async def login(req: LoginRequest):
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(User).where(User.email == req.email, User.is_active == True)
        )
        user = result.scalar_one_or_none()

    if not user or not verify_password(req.password, user.hashed_pw):
        raise HTTPException(status_code=401, detail="Email hoặc mật khẩu không đúng")

    token = _create_token(str(user.id), user.email)
    logger.info("User logged in | email=%s", user.email)
    return AuthResponse(
        access_token=token,
        user={"id": str(user.id), "email": user.email, "full_name": user.full_name or ""},
    )


@router.post("/register", response_model=AuthResponse)
async def register(req: RegisterRequest):
    # Log the incoming request for debugging
    logger.debug("Registration attempt | email=%s, full_name=%s", req.email, req.full_name)

    async with AsyncSessionLocal() as db:
        # Check if email exists
        result = await db.execute(select(User).where(User.email == req.email))
        if result.scalar_one_or_none():
            raise HTTPException(status_code=409, detail="Email đã tồn tại")

        new_user = User(
            id=uuid.uuid4(),
            email=req.email,
            hashed_pw=hash_password(req.password),
            full_name=req.full_name,
        )
        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)

    token = _create_token(str(new_user.id), new_user.email)
    logger.info("User registered | email=%s", new_user.email)
    return AuthResponse(
        access_token=token,
        user={"id": str(new_user.id), "email": new_user.email, "full_name": new_user.full_name or ""},
    )


@router.get("/me", response_model=UserInfo)
async def get_me(current_user: dict = Depends(get_current_user)):
    return UserInfo(
        id=current_user["user_id"],
        email=current_user["email"],
        full_name="",
    )
