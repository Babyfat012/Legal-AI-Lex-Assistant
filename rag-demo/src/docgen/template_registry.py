"""
Template Registry — Định nghĩa metadata cho từng loại đơn pháp lý.

Mỗi template gồm:
- Thông tin hiển thị (tên, mô tả)
- Path tới file .docx template
- Danh sách fields cần thu thập từ user
"""

import os
from dataclasses import dataclass, field
from typing import Literal

TEMPLATES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "templates"
)


@dataclass
class FieldDef:
    """Định nghĩa 1 field cần thu thập."""
    name: str                    # Key trong context dict (e.g. "ho_ten_vo")
    label: str                   # Tên hiển thị tiếng Việt
    label_en: str                # Tên hiển thị tiếng Anh
    placeholder: str             # Ví dụ giá trị
    required: bool = True        # Bắt buộc?
    field_type: Literal["text", "date", "textarea"] = "text"
    group: str = ""              # Nhóm hiển thị (e.g. "Thông tin bên vợ")
    group_en: str = ""


@dataclass
class TemplateDef:
    """Định nghĩa 1 loại đơn."""
    template_id: str
    display_name: str
    display_name_en: str
    description: str
    description_en: str
    template_filename: str       # Tên file trong templates/ dir
    fields: list[FieldDef]
    output_prefix: str           # Prefix tên file output

    @property
    def template_path(self) -> str:
        return os.path.join(TEMPLATES_DIR, self.template_filename)

    @property
    def required_fields(self) -> list[FieldDef]:
        return [f for f in self.fields if f.required]

    def to_dict(self) -> dict:
        """Serialize cho API response."""
        return {
            "template_id": self.template_id,
            "display_name": self.display_name,
            "display_name_en": self.display_name_en,
            "description": self.description,
            "description_en": self.description_en,
            "fields": [
                {
                    "name": f.name,
                    "label": f.label,
                    "label_en": f.label_en,
                    "placeholder": f.placeholder,
                    "required": f.required,
                    "field_type": f.field_type,
                    "group": f.group,
                    "group_en": f.group_en,
                }
                for f in self.fields
            ],
        }


# ---------------------------------------------------------------------------
# Template Definitions
# ---------------------------------------------------------------------------

_DON_LY_HON = TemplateDef(
    template_id="ly_hon",
    display_name="Đơn xin ly hôn",
    display_name_en="Divorce Application",
    description="Đơn yêu cầu Tòa án giải quyết ly hôn",
    description_en="Application requesting the Court to resolve a divorce",
    template_filename="don_ly_hon.docx",
    output_prefix="Don_Ly_Hon",
    fields=[
        # --- Tòa án ---
        FieldDef("toa_an", "Tòa án tiếp nhận", "Receiving Court",
                 "TAND quận Cầu Giấy, TP. Hà Nội", group="Thông tin Tòa án", group_en="Court Information"),
        # --- Người yêu cầu ---
        FieldDef("ho_ten_nguoi_yeu_cau", "Họ và tên người yêu cầu", "Applicant's full name",
                 "Nguyễn Văn A", group="Thông tin người yêu cầu", group_en="Applicant Information"),
        FieldDef("ngay_sinh_nguoi_yeu_cau", "Ngày sinh", "Date of birth",
                 "01/01/1990", field_type="text", group="Thông tin người yêu cầu", group_en="Applicant Information"),
        FieldDef("cccd_nguoi_yeu_cau", "Số CCCD/CMND", "ID Number",
                 "001090012345", group="Thông tin người yêu cầu", group_en="Applicant Information"),
        FieldDef("dia_chi_nguoi_yeu_cau", "Địa chỉ thường trú", "Permanent address",
                 "123 Đường ABC, Phường XYZ, Quận Cầu Giấy, Hà Nội",
                 group="Thông tin người yêu cầu", group_en="Applicant Information"),
        FieldDef("sdt_nguoi_yeu_cau", "Số điện thoại", "Phone number",
                 "0901234567", group="Thông tin người yêu cầu", group_en="Applicant Information"),
        # --- Bên kia ---
        FieldDef("ho_ten_ben_kia", "Họ và tên vợ/chồng", "Spouse's full name",
                 "Trần Thị B", group="Thông tin vợ/chồng", group_en="Spouse Information"),
        FieldDef("ngay_sinh_ben_kia", "Ngày sinh", "Date of birth",
                 "15/06/1992", field_type="text", group="Thông tin vợ/chồng", group_en="Spouse Information"),
        FieldDef("cccd_ben_kia", "Số CCCD/CMND", "ID Number",
                 "001092067890", group="Thông tin vợ/chồng", group_en="Spouse Information"),
        FieldDef("dia_chi_ben_kia", "Địa chỉ thường trú", "Permanent address",
                 "456 Đường DEF, Phường UVW, Quận Đống Đa, Hà Nội",
                 group="Thông tin vợ/chồng", group_en="Spouse Information"),
        # --- Nội dung ---
        FieldDef("ngay_ket_hon", "Ngày đăng ký kết hôn", "Marriage registration date",
                 "20/05/2015", group="Nội dung yêu cầu", group_en="Request Details"),
        FieldDef("noi_ket_hon", "Nơi đăng ký kết hôn", "Place of marriage registration",
                 "UBND phường XYZ, quận Cầu Giấy, Hà Nội",
                 group="Nội dung yêu cầu", group_en="Request Details"),
        FieldDef("ly_do", "Lý do ly hôn", "Reason for divorce",
                 "Vợ chồng mâu thuẫn kéo dài, không thể tiếp tục chung sống...",
                 field_type="textarea", group="Nội dung yêu cầu", group_en="Request Details"),
        FieldDef("con_chung", "Thông tin con chung (nếu có)", "Children information (if any)",
                 "Con: Nguyễn Văn C, sinh 10/03/2018. Yêu cầu: mẹ trực tiếp nuôi.",
                 field_type="textarea", required=False,
                 group="Nội dung yêu cầu", group_en="Request Details"),
        FieldDef("tai_san", "Tài sản chung (nếu có)", "Shared assets (if any)",
                 "Không có tài sản chung cần phân chia",
                 field_type="textarea", required=False,
                 group="Nội dung yêu cầu", group_en="Request Details"),
    ],
)

_DON_KHIEU_NAI = TemplateDef(
    template_id="khieu_nai",
    display_name="Đơn khiếu nại",
    display_name_en="Complaint Letter",
    description="Đơn khiếu nại quyết định hành chính, hành vi hành chính",
    description_en="Complaint against administrative decisions or actions",
    template_filename="don_khieu_nai.docx",
    output_prefix="Don_Khieu_Nai",
    fields=[
        FieldDef("co_quan", "Cơ quan/người bị khiếu nại", "Complained authority/person",
                 "UBND phường XYZ, quận Cầu Giấy, Hà Nội",
                 group="Thông tin tiếp nhận", group_en="Receiving Information"),
        FieldDef("ho_ten", "Họ và tên người khiếu nại", "Complainant's full name",
                 "Nguyễn Văn A", group="Thông tin người khiếu nại", group_en="Complainant Information"),
        FieldDef("ngay_sinh", "Ngày sinh", "Date of birth",
                 "01/01/1990", group="Thông tin người khiếu nại", group_en="Complainant Information"),
        FieldDef("cccd", "Số CCCD/CMND", "ID Number",
                 "001090012345", group="Thông tin người khiếu nại", group_en="Complainant Information"),
        FieldDef("dia_chi", "Địa chỉ thường trú", "Permanent address",
                 "123 Đường ABC, Phường XYZ, Quận Cầu Giấy, Hà Nội",
                 group="Thông tin người khiếu nại", group_en="Complainant Information"),
        FieldDef("sdt", "Số điện thoại", "Phone number",
                 "0901234567", group="Thông tin người khiếu nại", group_en="Complainant Information"),
        FieldDef("doi_tuong_khieu_nai", "Quyết định/hành vi bị khiếu nại",
                 "Decision/action being complained about",
                 "Quyết định số 123/QĐ-UBND ngày 01/01/2025 về việc...",
                 field_type="textarea", group="Nội dung khiếu nại", group_en="Complaint Content"),
        FieldDef("noi_dung", "Nội dung khiếu nại", "Complaint details",
                 "Quyết định trên không đúng quy định pháp luật vì...",
                 field_type="textarea", group="Nội dung khiếu nại", group_en="Complaint Content"),
        FieldDef("yeu_cau", "Yêu cầu giải quyết", "Resolution request",
                 "Hủy bỏ Quyết định số 123/QĐ-UBND và khôi phục quyền lợi hợp pháp...",
                 field_type="textarea", group="Nội dung khiếu nại", group_en="Complaint Content"),
    ],
)

_DON_KHOI_KIEN = TemplateDef(
    template_id="khoi_kien",
    display_name="Đơn khởi kiện",
    display_name_en="Lawsuit Petition",
    description="Đơn khởi kiện vụ án dân sự tại Tòa án",
    description_en="Civil lawsuit petition filed at Court",
    template_filename="don_khoi_kien.docx",
    output_prefix="Don_Khoi_Kien",
    fields=[
        FieldDef("toa_an", "Tòa án tiếp nhận", "Receiving Court",
                 "TAND quận Cầu Giấy, TP. Hà Nội",
                 group="Thông tin Tòa án", group_en="Court Information"),
        # --- Nguyên đơn ---
        FieldDef("ho_ten_nguyen_don", "Họ và tên nguyên đơn", "Plaintiff's full name",
                 "Nguyễn Văn A", group="Nguyên đơn", group_en="Plaintiff"),
        FieldDef("ngay_sinh_nguyen_don", "Ngày sinh", "Date of birth",
                 "01/01/1990", group="Nguyên đơn", group_en="Plaintiff"),
        FieldDef("cccd_nguyen_don", "Số CCCD/CMND", "ID Number",
                 "001090012345", group="Nguyên đơn", group_en="Plaintiff"),
        FieldDef("dia_chi_nguyen_don", "Địa chỉ thường trú", "Permanent address",
                 "123 Đường ABC, Phường XYZ, Quận Cầu Giấy, Hà Nội",
                 group="Nguyên đơn", group_en="Plaintiff"),
        FieldDef("sdt_nguyen_don", "Số điện thoại", "Phone number",
                 "0901234567", group="Nguyên đơn", group_en="Plaintiff"),
        # --- Bị đơn ---
        FieldDef("ho_ten_bi_don", "Họ và tên bị đơn", "Defendant's full name",
                 "Trần Thị B", group="Bị đơn", group_en="Defendant"),
        FieldDef("dia_chi_bi_don", "Địa chỉ bị đơn", "Defendant's address",
                 "456 Đường DEF, Phường UVW, Quận Đống Đa, Hà Nội",
                 group="Bị đơn", group_en="Defendant"),
        # --- Nội dung ---
        FieldDef("noi_dung_khoi_kien", "Nội dung khởi kiện và quá trình sự việc",
                 "Lawsuit details and sequence of events",
                 "Ngày ... tôi có ký hợp đồng... với bị đơn. Tuy nhiên bị đơn đã không thực hiện...",
                 field_type="textarea", group="Nội dung khởi kiện", group_en="Lawsuit Content"),
        FieldDef("yeu_cau_khoi_kien", "Yêu cầu Tòa án giải quyết",
                 "Request for Court resolution",
                 "Buộc bị đơn thực hiện nghĩa vụ... / bồi thường thiệt hại...",
                 field_type="textarea", group="Nội dung khởi kiện", group_en="Lawsuit Content"),
        FieldDef("chung_cu", "Tài liệu, chứng cứ kèm theo",
                 "Attached documents and evidence",
                 "Hợp đồng số..., Biên bản..., Ảnh chụp...",
                 field_type="textarea", required=False,
                 group="Nội dung khởi kiện", group_en="Lawsuit Content"),
    ],
)


# ---------------------------------------------------------------------------
# Registry — singleton lookup
# ---------------------------------------------------------------------------

TEMPLATE_REGISTRY: dict[str, TemplateDef] = {
    t.template_id: t
    for t in [_DON_LY_HON, _DON_KHIEU_NAI, _DON_KHOI_KIEN]
}


def get_template(template_id: str) -> TemplateDef | None:
    return TEMPLATE_REGISTRY.get(template_id)


def list_templates() -> list[dict]:
    return [t.to_dict() for t in TEMPLATE_REGISTRY.values()]
