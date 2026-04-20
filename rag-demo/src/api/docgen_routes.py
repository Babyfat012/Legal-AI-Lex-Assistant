"""
Document Generation API Routes.

Endpoints for intent detection, template listing, and DOCX generation.
"""

import base64
import uuid
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.auth_routes import get_current_user
from auth.database import AsyncSessionLocal, GeneratedDocument
from sqlalchemy import select
from docgen.template_registry import list_templates, get_template
from docgen.intent_detector import IntentDetector
from docgen.generator import DocumentGenerator
from core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Document Generation"])

# Singletons
_intent_detector = IntentDetector()
_doc_generator = DocumentGenerator()


# --- Schemas ---

class IntentDetectRequest(BaseModel):
    query: str = Field(..., min_length=1)


class IntentDetectResponse(BaseModel):
    intent: str
    template_id: str | None = None
    template_name: str | None = None
    confidence: float


class DocGenRequest(BaseModel):
    template_id: str = Field(..., min_length=1)
    fields: dict[str, str] = Field(default_factory=dict)


class DocGenResponse(BaseModel):
    filename: str
    content_base64: str
    mime_type: str = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


# --- Endpoints ---

@router.post("/docgen/detect-intent", response_model=IntentDetectResponse)
async def detect_intent(
    req: IntentDetectRequest,
    current_user: dict = Depends(get_current_user),
):
    """Detect if user wants to create a document or ask a legal question."""
    result = _intent_detector.detect(req.query)

    template_name = None
    if result["template_id"]:
        tpl = get_template(result["template_id"])
        if tpl:
            template_name = tpl.display_name

    return IntentDetectResponse(
        intent=result["intent"],
        template_id=result["template_id"],
        template_name=template_name,
        confidence=result["confidence"],
    )


@router.get("/docgen/templates")
async def get_templates(current_user: dict = Depends(get_current_user)):
    """List all available document templates with their field definitions."""
    return list_templates()


@router.get("/docgen/templates/{template_id}")
async def get_template_detail(
    template_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Get a specific template's field definitions."""
    tpl = get_template(template_id)
    if not tpl:
        raise HTTPException(status_code=404, detail=f"Template not found: {template_id}")
    return tpl.to_dict()


@router.get("/docgen/history")
async def get_document_history(
    current_user: dict = Depends(get_current_user),
):
    """Get the user's generated document history."""
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(GeneratedDocument)
            .where(GeneratedDocument.user_id == uuid.UUID(current_user["user_id"]))
            .order_by(GeneratedDocument.created_at.desc())
            .limit(50)  # Limit to 50 most recent documents
        )
        documents = result.scalars().all()

        return [
            {
                "id": str(doc.id),
                "template_id": doc.template_id,
                "template_name": doc.template_name,
                "filename": doc.filename,
                "created_at": doc.created_at.isoformat(),
                "field_values": doc.field_values
            }
            for doc in documents
        ]


@router.post("/docgen/generate", response_model=DocGenResponse)
async def generate_document(
    req: DocGenRequest,
    current_user: dict = Depends(get_current_user),
):
    """Generate a DOCX document from template + field data. Returns base64-encoded file."""
    try:
        file_bytes, filename = _doc_generator.render(req.template_id, req.fields)
        content_b64 = base64.b64encode(file_bytes).decode("utf-8")

        # Save to database
        async with AsyncSessionLocal() as db:
            db_document = GeneratedDocument(
                user_id=uuid.UUID(current_user["user_id"]),
                template_id=req.template_id,
                template_name=get_template(req.template_id).display_name,
                field_values=req.fields,
                filename=filename,
                file_content=content_b64,
                mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
            db.add(db_document)
            await db.commit()

        logger.info(
            "Document generated and stored | template=%s | user=%s | filename=%s",
            req.template_id, current_user["email"], filename,
        )
        return DocGenResponse(
            filename=filename,
            content_base64=content_b64,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Document generation failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
