from chainlit.data import BaseDataLayer
from chainlit.types import (
    Pagination, ThreadFilter, PaginatedResponse,
    ThreadDict, PageInfo
)
import chainlit as cl
from datetime import datetime
from sqlalchemy import select, delete
from sqlalchemy.dialects.postgresql import insert
from src.auth.database import AsyncSessionLocal, User, Conversation, Message
from src.auth.security import verify_password, hash_password
import uuid


class PostgreSQLDataLayer(BaseDataLayer):
    """
    Implement interface này để Chainlit tự động:
    1. Hiện sidebar "Lịch sử hội thoại"
    2. Render danh sách threads với title + date
    3. Cho phép click resume conversation
    4. Lưu message vào PostgreSQL sau mỗi turn
    """

    # ── USER MANAGEMENT ──────────────────────────────────────────────

    async def get_user(self, identifier: str) -> cl.PersistedUser | None:
        """Chainlit gọi khi cần load user theo email."""
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(User).where(User.email == identifier)
            )
            user = result.scalar_one_or_none()
            if user:
                return cl.PersistedUser(
                    id=str(user.id),
                    identifier=user.email,
                    metadata={"full_name": user.full_name},
                    createdAt=user.created_at.isoformat() if user.created_at else datetime.utcnow().isoformat()
                )
        return None

    async def create_user(self, user: cl.User) -> cl.PersistedUser | None:
        """Chainlit gọi khi user đăng ký lần đầu."""
        async with AsyncSessionLocal() as db:
            result = await db.execute(select(User).where(User.email == user.identifier))
            existing_user = result.scalar_one_or_none()
            if existing_user:
                return cl.PersistedUser(
                    id=str(existing_user.id),
                    identifier=existing_user.email,
                    createdAt=existing_user.created_at.isoformat() if existing_user.created_at else datetime.utcnow().isoformat()
                )
            
            new_user = User(
                email=user.identifier,
                hashed_pw=hash_password("changeme123"),  # User phải đổi sau
                full_name=user.metadata.get("full_name", "")
            )
            db.add(new_user)
            await db.commit()
            await db.refresh(new_user)
            return cl.PersistedUser(
                id=str(new_user.id),
                identifier=new_user.email,
                createdAt=new_user.created_at.isoformat() if new_user.created_at else datetime.utcnow().isoformat()
            )

    # ── THREAD (CONVERSATION) MANAGEMENT ─────────────────────────────

    async def list_threads(
        self,
        pagination: Pagination,
        filters: ThreadFilter
    ) -> PaginatedResponse[ThreadDict]:
        """
        Chainlit gọi để render sidebar "Lịch sử hội thoại".
        Return list conversations → Chainlit hiện lên sidebar tự động.
        """
        async with AsyncSessionLocal() as db:
            query = select(Conversation, User.email).outerjoin(
                User, Conversation.user_id == User.id
            ).where(
                Conversation.user_id == uuid.UUID(filters.userId)
            ).order_by(Conversation.updated_at.desc())

            if pagination.cursor:
                # Offset-based pagination (cursor = offset)
                query = query.offset(int(pagination.cursor))
            query = query.limit(pagination.first or 20)

            result = await db.execute(query)
            conversations_with_email = result.all()

        threads = [
            ThreadDict(
                id=str(row.Conversation.id),
                name=row.Conversation.title or "Hội thoại mới",
                createdAt=row.Conversation.created_at.isoformat() + "Z" if row.Conversation.created_at else datetime.utcnow().isoformat() + "Z",
                userId=str(row.Conversation.user_id) if row.Conversation.user_id else "",
                userIdentifier=row.email if row.email else "",
                tags=[],
                metadata={},
                steps=[],  # Steps sẽ được load lazy khi user click
            )
            for row in conversations_with_email
        ]

        return PaginatedResponse(
            data=threads,
            pageInfo=PageInfo(
                hasNextPage=len(threads) == (pagination.first or 20),
                startCursor="0",
                endCursor=str(len(threads))
            )
        )

    async def get_thread(self, thread_id: str) -> ThreadDict | None:
        """
        Chainlit gọi khi user click vào 1 conversation trong sidebar.
        Return ThreadDict với toàn bộ messages → Chainlit render lại lịch sử.
        """
        async with AsyncSessionLocal() as db:
            # Load conversation
            conv_result = await db.execute(
                select(Conversation, User.email)
                .outerjoin(User, Conversation.user_id == User.id)
                .where(Conversation.id == uuid.UUID(thread_id))
            )
            row = conv_result.one_or_none()
            if not row:
                return None
            conv = row.Conversation
            user_email = row.email

            # Load messages theo thứ tự
            msg_result = await db.execute(
                select(Message)
                .where(Message.conversation_id == conv.id)
                .order_by(Message.created_at.asc())
            )
            messages = msg_result.scalars().all()

        # Convert messages → Chainlit Step format
        steps = [
            {
                "id": str(m.id),
                "threadId": thread_id,
                "parentId": None,
                "name": m.role,          # "user" | "assistant"
                "type": "user_message" if m.role == "user" else "assistant_message",
                "output": m.content,
                "createdAt": m.created_at.isoformat() + "Z" if m.created_at else datetime.utcnow().isoformat() + "Z",
                "metadata": {"sources": m.sources or []},
            }
            for m in messages
        ]

        return ThreadDict(
            id=str(conv.id),
            name=conv.title or "Hội thoại mới",
            createdAt=conv.created_at.isoformat() + "Z" if conv.created_at else datetime.utcnow().isoformat() + "Z",
            userId=str(conv.user_id) if conv.user_id else "",
            userIdentifier=user_email if user_email else "",
            steps=steps,
        )

    async def create_thread(
        self,
        thread_id: str,
        name: str | None = None,
        user_id: str | None = None,
        metadata: dict | None = None,
        tags: list[str] | None = None,
    ) -> ThreadDict:
        """Chainlit gọi khi user bắt đầu conversation mới."""
        async with AsyncSessionLocal() as db:
            conv = Conversation(
                id=uuid.UUID(thread_id),
                user_id=uuid.UUID(user_id) if user_id else None,
                title=name or "Hội thoại mới",
            )
            db.add(conv)
            await db.commit()
        return ThreadDict(
            id=thread_id,
            name=name or "Hội thoại mới",
            createdAt=datetime.utcnow().isoformat(),
            userId=user_id or "",
            userIdentifier="",
            steps=[]
        )

    async def update_thread(
        self,
        thread_id: str,
        name: str | None = None,
        user_id: str | None = None,
        metadata: dict | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Cập nhật title conversation. Sử dụng UPSERT để tránh race condition."""
        if not name:
            return

        # Trim to a concise sidebar title (max 60 chars, cut at word boundary)
        trimmed = name.strip()
        if len(trimmed) > 60:
            cut = trimmed[:60].rsplit(" ", 1)[0]
            trimmed = cut + "…"

        async with AsyncSessionLocal() as db:
            # UPSERT: create conversation with title if not exists,
            # OR update title if it exists (even if create_step hasn't run yet)
            stmt = insert(Conversation).values(
                id=uuid.UUID(thread_id),
                user_id=uuid.UUID(user_id) if user_id else None,
                title=trimmed,
                updated_at=datetime.utcnow()
            ).on_conflict_do_update(
                index_elements=['id'],
                set_={'title': trimmed, 'updated_at': datetime.utcnow()}
            )
            await db.execute(stmt)
            await db.commit()

    async def delete_thread(self, thread_id: str) -> None:
        """User xóa conversation từ sidebar."""
        async with AsyncSessionLocal() as db:
            await db.execute(
                delete(Message).where(
                    Message.conversation_id == uuid.UUID(thread_id)
                )
            )
            await db.execute(
                delete(Conversation).where(Conversation.id == uuid.UUID(thread_id))
            )
            await db.commit()

    # ── MESSAGE (STEP) MANAGEMENT ─────────────────────────────────────

    async def create_step(self, step_dict: dict) -> None:
        """
        Chainlit tự động gọi sau mỗi message (user và assistant).
        Đây là nơi lưu toàn bộ lịch sử chat vào DB.
        """
        async with AsyncSessionLocal() as db:
            role = "user" if step_dict.get("type") == "user_message" else "assistant"
            content = step_dict.get("output", "")

            if not content:
                return  # Skip empty steps (internal Chainlit steps)

            # Make sure conversation exists
            # get user id from context
            try:
                user_id = cl.context.session.user.metadata.get("user_id") if cl.context.session.user else None
            except Exception:
                user_id = None
            
            # Check cl.User explicitly if user_id metadata isn't set
            if not user_id and cl.context.session.user and cl.context.session.user.identifier:
                try:
                    u_res = await db.execute(select(User).where(User.email == cl.context.session.user.identifier))
                    u_match = u_res.scalar_one_or_none()
                    if u_match: user_id = str(u_match.id)
                except Exception:
                    pass

            from sqlalchemy import text
            stmt = insert(Conversation).values(
                id=uuid.UUID(step_dict["threadId"]),
                user_id=uuid.UUID(user_id) if user_id else None,
                title="Hội thoại mới",
                updated_at=datetime.utcnow()
            ).on_conflict_do_update(
                index_elements=['id'],
                set_={
                    'updated_at': datetime.utcnow(),
                    # Only overwrite title with default if title hasn't been set yet
                    'title': text(
                        "CASE WHEN conversations.title = 'H\u1ed9i tho\u1ea1i m\u1edbi' "
                        "OR conversations.title IS NULL "
                        "THEN 'H\u1ed9i tho\u1ea1i m\u1edbi' "
                        "ELSE conversations.title END"
                    ),
                }
            )
            await db.execute(stmt)

            msg = Message(
                id=uuid.UUID(step_dict["id"]) if step_dict.get("id") else uuid.uuid4(),
                conversation_id=uuid.UUID(step_dict["threadId"]),
                role=role,
                content=content,
                sources=step_dict.get("metadata", {}).get("sources", []),
            )
            db.add(msg)

            await db.commit()

    async def update_step(self, step_dict: dict) -> None:
        """Cập nhật message (ví dụ: streaming hoàn tất)."""
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(Message).where(Message.id == uuid.UUID(step_dict["id"]))
            )
            msg = result.scalar_one_or_none()
            if msg:
                msg.content = step_dict.get("output", msg.content)
                await db.commit()

    async def delete_step(self, step_id: str) -> None:
        async with AsyncSessionLocal() as db:
            await db.execute(
                delete(Message).where(Message.id == uuid.UUID(step_id))
            )
            await db.commit()

    # ── ELEMENT MANAGEMENT (file attachments, etc.) ───────────────────

    async def create_element(self, element: dict) -> None:
        pass  # Phase 2: implement nếu cần lưu file attachments

    async def get_element(self, thread_id: str, element_id: str) -> dict | None:
        return None

    async def delete_element(self, element_id: str, thread_id: str | None = None) -> None:
        pass

    async def get_thread_author(self, thread_id: str) -> str:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(User.email)
                .join(Conversation, Conversation.user_id == User.id)
                .where(Conversation.id == uuid.UUID(thread_id))
            )
            email = result.scalar_one_or_none()
            if email:
                return email
        return ""

    async def delete_feedback(self, feedback_id: str) -> bool:
        return True

    async def upsert_feedback(self, feedback: dict) -> str:
        return ""

    async def build_debug_url(self) -> str:
        return ""

    async def close(self) -> None:
        pass

    async def get_favorite_steps(self, user_id: str) -> list[dict]:
        return []
