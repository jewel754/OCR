"""
schemas.py — Pydantic V2 data models for the MSME GST Invoice Engine.

Hierarchy:
  RawExtractionResult          ← raw VLM / OCR output (pre-validation)
    └─ RawLineItem

  Invoice                      ← validated, corrected invoice
    ├─ SupplierInfo
    ├─ BuyerInfo
    ├─ LineItem (list)
    │    └─ TaxBreakdown
    ├─ InvoiceTotals
    └─ BoundingBoxCollection   ← per-field bounding boxes for Streamlit overlay

  ValidationResult             ← gatekeeper output with structured error list
    └─ ValidationError (list)

  TallyPushResult              ← response from TallyPrime HTTP push
"""

from __future__ import annotations

import re
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    ConfigDict,
)


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────

class DocumentType(str, Enum):
    DIGITAL_PDF = "digital_pdf"      # text-layer PDF → Docling
    SCANNED_PDF = "scanned_pdf"      # image-only PDF → OpenCV + VLM
    IMAGE = "image"                  # JPG / PNG → OpenCV + VLM
    UNKNOWN = "unknown"


class TransactionType(str, Enum):
    INTRA_STATE = "intra_state"      # CGST + SGST
    INTER_STATE = "inter_state"      # IGST only
    EXPORT = "export"                # zero-rated


class ValidationSeverity(str, Enum):
    ERROR = "error"       # blocks Tally push
    WARNING = "warning"   # surfaced to human but doesn't block
    INFO = "info"         # informational note


class ExtractionStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"           # some fields missing; DoubleTake attempted
    DOUBLETAKE_RECOVERED = "doubletake_recovered"
    FAILED = "failed"


# ─────────────────────────────────────────────────────────────────────────────
# Sub-models
# ─────────────────────────────────────────────────────────────────────────────

class BoundingBox(BaseModel):
    """
    Normalised bounding box (0.0–1.0 relative to page dimensions).
    Origin = top-left corner.
    """
    model_config = ConfigDict(frozen=True)

    x: float = Field(..., ge=0.0, le=1.0, description="Left edge (normalised)")
    y: float = Field(..., ge=0.0, le=1.0, description="Top edge (normalised)")
    w: float = Field(..., ge=0.0, le=1.0, description="Width (normalised)")
    h: float = Field(..., ge=0.0, le=1.0, description="Height (normalised)")
    page: int = Field(default=1, ge=1, description="1-based page number")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    def to_pixel(self, img_width: int, img_height: int) -> tuple[int, int, int, int]:
        """Convert normalised coords to absolute pixel coordinates (x, y, w, h)."""
        return (
            int(self.x * img_width),
            int(self.y * img_height),
            int(self.w * img_width),
            int(self.h * img_height),
        )


class BoundingBoxCollection(BaseModel):
    """Maps each invoice field name to its bounding box."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    fields: dict[str, BoundingBox] = Field(default_factory=dict)

    def get(self, field_name: str) -> BoundingBox | None:
        return self.fields.get(field_name)

    def add(self, field_name: str, bbox: BoundingBox) -> None:
        self.fields[field_name] = bbox


class TaxBreakdown(BaseModel):
    """
    GST components for a single line item or the invoice total.
    All amounts are in INR.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    gst_rate: float = Field(..., ge=0.0, le=28.0, description="Applicable GST rate %")
    taxable_amount: float = Field(..., ge=0.0)

    # Intra-state fields (CGST + SGST)
    cgst_rate: float = Field(default=0.0, ge=0.0)
    cgst_amount: float = Field(default=0.0, ge=0.0)
    sgst_rate: float = Field(default=0.0, ge=0.0)
    sgst_amount: float = Field(default=0.0, ge=0.0)

    # Inter-state field (IGST)
    igst_rate: float = Field(default=0.0, ge=0.0)
    igst_amount: float = Field(default=0.0, ge=0.0)

    @property
    def total_tax(self) -> float:
        return round(self.cgst_amount + self.sgst_amount + self.igst_amount, 2)

    @property
    def line_total(self) -> float:
        return round(self.taxable_amount + self.total_tax, 2)


class LineItem(BaseModel):
    """A single line in the invoice."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    sl_no: int = Field(..., ge=1)
    description: str = Field(..., min_length=1)
    hsn_sac_code: str = Field(default="")
    quantity: float = Field(..., gt=0.0)
    unit: str = Field(default="NOS")
    unit_price: float = Field(..., gt=0.0)
    discount_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    discount_amount: float = Field(default=0.0, ge=0.0)
    taxable_amount: float = Field(..., ge=0.0)
    tax: TaxBreakdown

    # Reconciliation fields (filled by validator)
    matched_sku_code: str = Field(default="")
    matched_sku_name: str = Field(default="")
    sku_match_score: float = Field(default=0.0)
    po_rate_deviation_pct: float = Field(default=0.0)

    # Bounding box for this row
    bbox: BoundingBox | None = Field(default=None)

    @property
    def computed_taxable_amount(self) -> float:
        gross = round(self.quantity * self.unit_price, 2)
        disc = round(gross * self.discount_pct / 100, 2) if self.discount_pct else self.discount_amount
        return round(gross - disc, 2)

    @property
    def line_total(self) -> float:
        return round(self.taxable_amount + self.tax.total_tax, 2)


class SupplierInfo(BaseModel):
    """Supplier (vendor) details extracted from invoice header."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(default="")
    gstin: str = Field(default="")
    address: str = Field(default="")
    state_code: str = Field(default="")
    pan: str = Field(default="")
    email: str = Field(default="")
    phone: str = Field(default="")

    @field_validator("gstin")
    @classmethod
    def normalise_gstin(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("pan")
    @classmethod
    def normalise_pan(cls, v: str) -> str:
        return v.strip().upper()


class BuyerInfo(BaseModel):
    """Buyer (recipient) details — typically our company."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(default="")
    gstin: str = Field(default="")
    address: str = Field(default="")
    state_code: str = Field(default="")

    @field_validator("gstin")
    @classmethod
    def normalise_gstin(cls, v: str) -> str:
        return v.strip().upper()


class InvoiceTotals(BaseModel):
    """Extracted and computed invoice-level totals."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    subtotal: float = Field(default=0.0, ge=0.0, description="Sum of taxable amounts")
    total_cgst: float = Field(default=0.0, ge=0.0)
    total_sgst: float = Field(default=0.0, ge=0.0)
    total_igst: float = Field(default=0.0, ge=0.0)
    total_tax: float = Field(default=0.0, ge=0.0)
    round_off: float = Field(default=0.0, description="Rounding adjustment (can be negative)")
    grand_total: float = Field(default=0.0, ge=0.0)
    grand_total_in_words: str = Field(default="")

    # Computed check fields (set by validator)
    computed_subtotal: float = Field(default=0.0)
    computed_total_tax: float = Field(default=0.0)
    computed_grand_total: float = Field(default=0.0)

    @property
    def subtotal_discrepancy(self) -> float:
        return round(abs(self.subtotal - self.computed_subtotal), 2)

    @property
    def grand_total_discrepancy(self) -> float:
        return round(abs(self.grand_total - self.computed_grand_total), 2)


# ─────────────────────────────────────────────────────────────────────────────
# Core Invoice model
# ─────────────────────────────────────────────────────────────────────────────

class Invoice(BaseModel):
    """
    Fully parsed and validated GST Invoice.
    This is the central data object passed between all pipeline stages.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Document metadata
    invoice_id: str = Field(default="", description="Unique pipeline-assigned ID")
    source_filename: str = Field(default="")
    document_type: DocumentType = Field(default=DocumentType.UNKNOWN)
    extraction_status: ExtractionStatus = Field(default=ExtractionStatus.FAILED)
    page_count: int = Field(default=1, ge=1)

    # Invoice header
    invoice_number: str = Field(default="")
    invoice_date: str = Field(default="", description="ISO-8601 date string YYYY-MM-DD")
    po_number: str = Field(default="", description="Referenced Purchase Order number")
    place_of_supply: str = Field(default="", description="State code or state name")
    transaction_type: TransactionType = Field(default=TransactionType.INTRA_STATE)

    # Parties
    supplier: SupplierInfo = Field(default_factory=SupplierInfo)
    buyer: BuyerInfo = Field(default_factory=BuyerInfo)

    # Line items
    line_items: list[LineItem] = Field(default_factory=list)

    # Totals
    totals: InvoiceTotals = Field(default_factory=InvoiceTotals)

    # Bounding boxes (for Streamlit overlay)
    bboxes: BoundingBoxCollection = Field(default_factory=BoundingBoxCollection)

    # Pipeline metadata
    raw_text: str = Field(default="", description="Full raw OCR / Docling text dump")
    doubletake_triggered: bool = Field(default=False)
    doubletake_rows_recovered: int = Field(default=0)
    vlm_model_used: str = Field(default="demo_mock")
    processing_time_ms: float = Field(default=0.0)

    @property
    def line_item_count(self) -> int:
        return len(self.line_items)

    @property
    def is_intra_state(self) -> bool:
        return self.transaction_type == TransactionType.INTRA_STATE

    def recompute_totals(self) -> None:
        """
        Recompute all aggregate totals from line items.
        Call this after adding / modifying line items.
        """
        subtotal = sum(li.taxable_amount for li in self.line_items)
        cgst = sum(li.tax.cgst_amount for li in self.line_items)
        sgst = sum(li.tax.sgst_amount for li in self.line_items)
        igst = sum(li.tax.igst_amount for li in self.line_items)

        self.totals.computed_subtotal = round(subtotal, 2)
        self.totals.computed_total_tax = round(cgst + sgst + igst, 2)
        self.totals.computed_grand_total = round(subtotal + cgst + sgst + igst, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Raw Extraction models (pre-validation, tolerant of dirty data)
# ─────────────────────────────────────────────────────────────────────────────

class RawLineItem(BaseModel):
    """
    Loosely-typed line item as returned by VLM / OCR.
    All fields are strings to avoid parse failures on dirty data.
    """
    model_config = ConfigDict(extra="allow")

    sl_no: str = Field(default="")
    description: str = Field(default="")
    hsn_sac_code: str = Field(default="")
    quantity: str = Field(default="")
    unit: str = Field(default="")
    unit_price: str = Field(default="")
    discount_pct: str = Field(default="0")
    discount_amount: str = Field(default="0")
    taxable_amount: str = Field(default="")
    gst_rate: str = Field(default="")
    cgst_rate: str = Field(default="")
    cgst_amount: str = Field(default="")
    sgst_rate: str = Field(default="")
    sgst_amount: str = Field(default="")
    igst_rate: str = Field(default="")
    igst_amount: str = Field(default="")
    line_total: str = Field(default="")


class RawExtractionResult(BaseModel):
    """
    Raw, unvalidated extraction output from Docling / VLM / OCR.
    Fields are lenient strings; the validator converts them to typed Invoice.
    """
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    source_filename: str = Field(default="")
    document_type: str = Field(default="unknown")
    extraction_status: str = Field(default="failed")
    vlm_model_used: str = Field(default="demo_mock")
    page_count: int = Field(default=1)
    raw_text: str = Field(default="")

    invoice_number: str = Field(default="")
    invoice_date: str = Field(default="")
    po_number: str = Field(default="")
    place_of_supply: str = Field(default="")

    supplier_name: str = Field(default="")
    supplier_gstin: str = Field(default="")
    supplier_address: str = Field(default="")
    supplier_state_code: str = Field(default="")
    supplier_pan: str = Field(default="")

    buyer_name: str = Field(default="")
    buyer_gstin: str = Field(default="")
    buyer_address: str = Field(default="")
    buyer_state_code: str = Field(default="")

    line_items: list[RawLineItem] = Field(default_factory=list)

    subtotal: str = Field(default="")
    total_cgst: str = Field(default="")
    total_sgst: str = Field(default="")
    total_igst: str = Field(default="")
    total_tax: str = Field(default="")
    round_off: str = Field(default="0")
    grand_total: str = Field(default="")
    grand_total_in_words: str = Field(default="")

    # Bounding boxes keyed by field name
    bboxes: dict[str, dict[str, Any]] = Field(default_factory=dict)

    # DoubleTake metadata
    doubletake_triggered: bool = Field(default=False)
    doubletake_rows_recovered: int = Field(default=0)
    processing_time_ms: float = Field(default=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Validation result models
# ─────────────────────────────────────────────────────────────────────────────

class ValidationError(BaseModel):
    """A single validation finding."""
    model_config = ConfigDict(frozen=True)

    code: str = Field(..., description="Machine-readable error code e.g. 'MATH_001'")
    severity: ValidationSeverity
    field: str = Field(default="", description="Dot-path to the offending field")
    expected: str = Field(default="", description="Expected value or range")
    found: str = Field(default="", description="Actual extracted value")
    message: str = Field(..., description="Human-readable description")
    auto_corrected: bool = Field(default=False, description="Was this error automatically corrected?")
    correction_applied: str = Field(default="", description="Description of the correction")

    @property
    def blocks_push(self) -> bool:
        """ERROR-level findings block the Tally push; warnings do not."""
        return self.severity == ValidationSeverity.ERROR and not self.auto_corrected


class ValidationResult(BaseModel):
    """
    Output from validator.py — attached to the Invoice after gate-keeping.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    invoice_id: str = Field(default="")
    is_valid: bool = Field(default=False, description="True = safe to push to Tally")
    errors: list[ValidationError] = Field(default_factory=list)
    corrected_invoice: Invoice | None = Field(default=None, description="Invoice with auto-corrections applied")

    @property
    def error_count(self) -> int:
        return sum(1 for e in self.errors if e.severity == ValidationSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for e in self.errors if e.severity == ValidationSeverity.WARNING)

    @property
    def blocking_errors(self) -> list[ValidationError]:
        return [e for e in self.errors if e.blocks_push]

    @property
    def auto_corrected_count(self) -> int:
        return sum(1 for e in self.errors if e.auto_corrected)

    @property
    def summary(self) -> str:
        status = "✅ VALID" if self.is_valid else "❌ INVALID"
        return (
            f"{status} | Errors: {self.error_count} | "
            f"Warnings: {self.warning_count} | "
            f"Auto-corrected: {self.auto_corrected_count}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tally Push result
# ─────────────────────────────────────────────────────────────────────────────

class TallyPushResult(BaseModel):
    """Result of pushing a validated invoice XML to TallyPrime."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    invoice_id: str = Field(default="")
    success: bool = Field(default=False)
    tally_response_xml: str = Field(default="")
    tally_voucher_number: str = Field(default="", description="Voucher number assigned by Tally")
    error_message: str = Field(default="")
    http_status_code: int = Field(default=0)
    pushed_at: str = Field(default="", description="ISO-8601 timestamp")
    xml_payload_preview: str = Field(default="", description="First 500 chars of sent XML")


# ─────────────────────────────────────────────────────────────────────────────
# Master data models (used by mock_data.py and validator.py)
# ─────────────────────────────────────────────────────────────────────────────

class PartyMaster(BaseModel):
    """Registered party (supplier/customer) in the master ledger."""
    model_config = ConfigDict(frozen=False)

    party_id: str
    name: str
    gstin: str
    state_code: str
    address: str
    pan: str = Field(default="")
    email: str = Field(default="")
    phone: str = Field(default="")
    is_active: bool = Field(default=True)

    @field_validator("gstin")
    @classmethod
    def normalise_gstin(cls, v: str) -> str:
        return v.strip().upper()


class SKUMaster(BaseModel):
    """Stock-Keeping Unit in the item master."""
    model_config = ConfigDict(frozen=False)

    sku_code: str
    name: str
    aliases: list[str] = Field(default_factory=list, description="Alternate names / descriptions")
    hsn_sac_code: str = Field(default="")
    unit: str = Field(default="NOS")
    gst_rate: float = Field(default=18.0, ge=0.0, le=28.0)
    standard_rate: float = Field(default=0.0, ge=0.0, description="Standard purchase price (₹)")
    is_active: bool = Field(default=True)

    @property
    def all_names(self) -> list[str]:
        """All searchable names for fuzzy matching."""
        return [self.name] + self.aliases


class PurchaseOrder(BaseModel):
    """Simplified PO record for rate-deviation check."""
    model_config = ConfigDict(frozen=False)

    po_number: str
    supplier_gstin: str
    items: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of {sku_code, qty, unit_price} dicts"
    )
    po_date: str = Field(default="")
    is_open: bool = Field(default=True)
