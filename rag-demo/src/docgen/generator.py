"""
Document Generator — Render DOCX từ template + user data.

Sử dụng docxtpl (Jinja2-based) để render file .docx template
với context data thu thập từ user.
"""

import os
from io import BytesIO
from datetime import datetime
from docxtpl import DocxTemplate
from docgen.template_registry import TEMPLATE_REGISTRY, get_template
from core.logger import get_logger

logger = get_logger(__name__)


class DocumentGenerator:
    """Render DOCX file từ template + context data."""

    def render(self, template_id: str, fields: dict) -> tuple[bytes, str]:
        """
        Render a DOCX template with the given fields.

        Args:
            template_id: ID of the template to use
            fields: Dict of field_name -> value

        Returns:
            Tuple of (file_bytes, filename)

        Raises:
            ValueError: If template_id not found or template file missing
        """
        template_def = get_template(template_id)
        if not template_def:
            raise ValueError(f"Template not found: {template_id}")

        template_path = template_def.template_path
        if not os.path.exists(template_path):
            raise ValueError(f"Template file not found: {template_path}")

        logger.info("Rendering document | template=%s | fields=%d", template_id, len(fields))

        # Add auto-generated fields
        context = dict(fields)
        now = datetime.now()
        context.setdefault("ngay_lam_don", now.strftime("%d/%m/%Y"))
        context.setdefault("ngay", str(now.day))
        context.setdefault("thang", str(now.month))
        context.setdefault("nam", str(now.year))

        # Render template
        doc = DocxTemplate(template_path)
        doc.render(context)

        # Save to bytes
        buffer = BytesIO()
        doc.save(buffer)
        buffer.seek(0)

        timestamp = now.strftime("%Y%m%d_%H%M%S")
        filename = f"{template_def.output_prefix}_{timestamp}.docx"

        logger.info("Document rendered successfully | filename=%s", filename)
        return buffer.getvalue(), filename
