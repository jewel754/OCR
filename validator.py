"""
validator.py — Deterministic GST Invoice Validation Gatekeeper.

Validation checks performed (in order):
  MATH checks
    MATH-001  Line-item taxable = qty × unit_price − discount
    MATH-002  CGSTamt = taxable × cgst_rate / 100  (intra-state only)
    MATH-003  SGSTamt = taxable × sgst_rate / 100  (intra-state only)
    MATH-004  IGSTamt = taxable × igst_rate / 100  (inter-state only)
    MATH-005  Invoice subtotal = Σ taxable_amounts
    MATH-006  Total tax = Σ (cgst + sgst + igst) across all lines
    MATH-007  Grand total = subtotal + total_tax + round_off

  GST STATE checks
    GST-001   GSTIN structural format (15 chars, regex)
    GST-002   GSTIN checksum validity (Govt modulus-36 algo)
    GST-003   Supplier state_code matches first 2 chars of GSTIN
    GST-004   CGST+SGST used for intra-state; IGST used for inter-state
    GST-005   GST rate must be in the valid rate schedule
    GST-006   CGST rate = SGST rate = GST rate / 2 (intra-state)

  MASTER RECONCILIATION
    SKU-001   Each line description fuzzy-matched against SKU master
    SKU-002   HSN/SAC code cross-check vs matched SKU master entry
    PO-001    PO number found in open PO master (warning if not found)
    PO-002    Billed rate > PO rate x 1.005 = ERROR (blocking overbilling); billed < PO by > cfg % = WARNING

  STRUCTURE
    STRUCT-001  Invoice number non-empty
    STRUCT-002  Invoice date parseable and not in future
    STRUCT-003  At least one line item present
    STRUCT-004  Buyer GSTIN matches configured company GSTIN

Auto-correction logic (applied before returning corrected_invoice):
  • MATH-001:  Recompute taxable from qty/price if deviation ≤ ₹2
  • MATH-002/3/4: Recompute tax amounts from taxable and rates
  • MATH-005/6/7: Recompute aggregate totals from corrected lines
  • GST-003:  Derive state_code from GSTIN[:2]
"""

from __future__ import annotations

import copy
import re
import uuid
from datetime import date, datetime
from typing import Any

from rapidfuzz import fuzz, process

from config import (
    cfg,
    INDIAN_STATE_CODES,
    VALID_GST_RATES,
    validate_gstin_checksum,
    is_inter_state,
    gstin_state_code,
)
from schemas import (
    Invoice,
    LineItem,
    SupplierInfo,
    BuyerInfo,
    InvoiceTotals,
    TaxBreakdown,
    BoundingBox,
    BoundingBoxCollection,
    DocumentType,
    TransactionType,
    ExtractionStatus,
    ValidationError,
    ValidationResult,
    ValidationSeverity,
    RawExtractionResult,
    RawLineItem,
    PartyMaster,
    SKUMaster,
    PurchaseOrder,
)

# ─────────────────────────────────────────────────────────────────────────────
# Regex patterns
# ─────────────────────────────────────────────────────────────────────────────

_GSTIN_RE = re.compile(
    r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$"
)
_PAN_RE = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]{1}$")
_DATE_FORMATS = ["%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y", "%d %b %Y", "%d %B %Y"]


# ─────────────────────────────────────────────────────────────────────────────
# Type-coercion helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(value: Any, default: float = 0.0) -> float:
    """
    Coerce string/int/float to float, stripping ₹, commas, spaces.
    Returns default on failure.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        return default
    cleaned = str(value).replace(",", "").replace("₹", "").replace(" ", "").strip()
    if cleaned in ("", "-", "N/A", "NA", "nil", "Nil"):
        return default
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    """Coerce to int, stripping non-numeric characters."""
    try:
        return int(_safe_float(value, float(default)))
    except (ValueError, TypeError):
        return default


def _parse_date(date_str: str) -> date | None:
    """Try multiple date formats; return None on failure."""
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(date_str.strip(), fmt).date()
        except (ValueError, AttributeError):
            continue
    return None


def _normalise_gstin(raw: str) -> str:
    return raw.strip().upper().replace(" ", "").replace("-", "")


# ─────────────────────────────────────────────────────────────────────────────
# Raw → typed conversion
# ─────────────────────────────────────────────────────────────────────────────

def _raw_to_invoice(raw: RawExtractionResult) -> Invoice:
    """
    Convert a RawExtractionResult (all strings) into a typed Invoice.
    Does NOT validate — just type-casts and structures the data.
    """
    supplier_gstin = _normalise_gstin(raw.supplier_gstin)
    buyer_gstin    = _normalise_gstin(raw.buyer_gstin)
    supplier_state = raw.supplier_state_code or gstin_state_code(supplier_gstin)
    buyer_state    = raw.buyer_state_code or gstin_state_code(buyer_gstin) or cfg.COMPANY_STATE_CODE

    inter_state_flag = is_inter_state(supplier_gstin, buyer_gstin) if (supplier_gstin and buyer_gstin) else False
    tx_type = TransactionType.INTER_STATE if inter_state_flag else TransactionType.INTRA_STATE

    # Parse line items
    typed_items: list[LineItem] = []
    for idx, raw_item in enumerate(raw.line_items):
        qty     = _safe_float(raw_item.quantity, 1.0)
        price   = _safe_float(raw_item.unit_price)
        disc_pct = _safe_float(raw_item.discount_pct)
        disc_amt = _safe_float(raw_item.discount_amount)
        taxable  = _safe_float(raw_item.taxable_amount)
        gst_rate = _safe_float(raw_item.gst_rate, 18.0)
        half     = round(gst_rate / 2, 1)

        cgst_r   = _safe_float(raw_item.cgst_rate, half if not inter_state_flag else 0.0)
        cgst_a   = _safe_float(raw_item.cgst_amount)
        sgst_r   = _safe_float(raw_item.sgst_rate, half if not inter_state_flag else 0.0)
        sgst_a   = _safe_float(raw_item.sgst_amount)
        igst_r   = _safe_float(raw_item.igst_rate, gst_rate if inter_state_flag else 0.0)
        igst_a   = _safe_float(raw_item.igst_amount)

        tax = TaxBreakdown(
            gst_rate=gst_rate,
            taxable_amount=taxable,
            cgst_rate=cgst_r,
            cgst_amount=cgst_a,
            sgst_rate=sgst_r,
            sgst_amount=sgst_a,
            igst_rate=igst_r,
            igst_amount=igst_a,
        )

        # Try to get bounding box for this line
        bbox_key = f"line_item_{idx+1}"
        bbox_data = raw.bboxes.get(bbox_key)
        bbox = None
        if bbox_data:
            try:
                bbox = BoundingBox(**bbox_data)
            except Exception:
                bbox = None

        typed_items.append(
            LineItem(
                sl_no=_safe_int(raw_item.sl_no, idx + 1),
                description=raw_item.description.strip() or f"Item {idx+1}",
                hsn_sac_code=raw_item.hsn_sac_code.strip(),
                quantity=qty,
                unit=raw_item.unit.strip() or "NOS",
                unit_price=price,
                discount_pct=disc_pct,
                discount_amount=disc_amt,
                taxable_amount=taxable,
                tax=tax,
                bbox=bbox,
            )
        )

    # Parse bounding boxes for header/footer fields
    bbox_coll = BoundingBoxCollection()
    for field_name, bbox_data in raw.bboxes.items():
        if not field_name.startswith("line_item_"):
            try:
                bbox_coll.add(field_name, BoundingBox(**bbox_data))
            except Exception:
                pass

    totals = InvoiceTotals(
        subtotal=_safe_float(raw.subtotal),
        total_cgst=_safe_float(raw.total_cgst),
        total_sgst=_safe_float(raw.total_sgst),
        total_igst=_safe_float(raw.total_igst),
        total_tax=_safe_float(raw.total_tax),
        round_off=_safe_float(raw.round_off),
        grand_total=_safe_float(raw.grand_total),
        grand_total_in_words=raw.grand_total_in_words,
    )

    return Invoice(
        invoice_id=str(uuid.uuid4())[:8].upper(),
        source_filename=raw.source_filename,
        document_type=DocumentType(raw.document_type) if raw.document_type in DocumentType.__members__.values() else DocumentType.UNKNOWN,
        extraction_status=ExtractionStatus(raw.extraction_status) if raw.extraction_status in ExtractionStatus.__members__.values() else ExtractionStatus.PARTIAL,
        page_count=raw.page_count,
        invoice_number=raw.invoice_number.strip(),
        invoice_date=raw.invoice_date.strip(),
        po_number=raw.po_number.strip(),
        place_of_supply=raw.place_of_supply.strip(),
        transaction_type=tx_type,
        supplier=SupplierInfo(
            name=raw.supplier_name,
            gstin=supplier_gstin,
            address=raw.supplier_address,
            state_code=supplier_state,
            pan=raw.supplier_pan,
        ),
        buyer=BuyerInfo(
            name=raw.buyer_name,
            gstin=buyer_gstin,
            address=raw.buyer_address,
            state_code=buyer_state,
        ),
        line_items=typed_items,
        totals=totals,
        bboxes=bbox_coll,
        raw_text=raw.raw_text,
        doubletake_triggered=raw.doubletake_triggered,
        doubletake_rows_recovered=raw.doubletake_rows_recovered,
        vlm_model_used=raw.vlm_model_used,
        processing_time_ms=raw.processing_time_ms,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Individual check functions
# ─────────────────────────────────────────────────────────────────────────────

def _check_struct(inv: Invoice) -> list[ValidationError]:
    errors: list[ValidationError] = []

    # STRUCT-001: Invoice number
    if not inv.invoice_number:
        errors.append(ValidationError(
            code="STRUCT-001",
            severity=ValidationSeverity.ERROR,
            field="invoice_number",
            expected="non-empty string",
            found="(empty)",
            message="Invoice number is missing.",
        ))

    # STRUCT-002: Invoice date
    parsed_date = _parse_date(inv.invoice_date)
    if not parsed_date:
        errors.append(ValidationError(
            code="STRUCT-002",
            severity=ValidationSeverity.ERROR,
            field="invoice_date",
            expected="valid date in YYYY-MM-DD or DD-MM-YYYY format",
            found=repr(inv.invoice_date),
            message=f"Invoice date '{inv.invoice_date}' cannot be parsed.",
        ))
    elif parsed_date > date.today():
        errors.append(ValidationError(
            code="STRUCT-002B",
            severity=ValidationSeverity.WARNING,
            field="invoice_date",
            expected=f"<= today ({date.today().isoformat()})",
            found=parsed_date.isoformat(),
            message="Invoice date is in the future — possible OCR misread.",
        ))

    # STRUCT-003: At least one line item
    if not inv.line_items:
        errors.append(ValidationError(
            code="STRUCT-003",
            severity=ValidationSeverity.ERROR,
            field="line_items",
            expected=">= 1 line item",
            found="0 items",
            message="No line items extracted. Invoice cannot be processed.",
        ))

    # STRUCT-004: Buyer GSTIN matches company
    expected_buyer = cfg.COMPANY_GSTIN.upper()
    if inv.buyer.gstin and inv.buyer.gstin != expected_buyer:
        errors.append(ValidationError(
            code="STRUCT-004",
            severity=ValidationSeverity.WARNING,
            field="buyer.gstin",
            expected=expected_buyer,
            found=inv.buyer.gstin,
            message=(
                f"Buyer GSTIN '{inv.buyer.gstin}' does not match "
                f"configured company GSTIN '{expected_buyer}'."
            ),
        ))

    return errors


def _check_gst(inv: Invoice) -> tuple[list[ValidationError], Invoice]:
    """
    Run all GST/GSTIN checks. Returns errors and (possibly corrected) invoice.
    """
    errors: list[ValidationError] = []
    corrected = copy.deepcopy(inv)

    supplier_gstin = inv.supplier.gstin

    # GST-001: Structural regex
    if supplier_gstin and not _GSTIN_RE.match(supplier_gstin):
        # Attempt auto-correction: strip non-alphanum and pad/truncate
        cleaned = re.sub(r"[^A-Z0-9]", "", supplier_gstin)
        if len(cleaned) == 15 and _GSTIN_RE.match(cleaned):
            corrected.supplier.gstin = cleaned
            errors.append(ValidationError(
                code="GST-001",
                severity=ValidationSeverity.WARNING,
                field="supplier.gstin",
                expected="15-char GSTIN format",
                found=supplier_gstin,
                message=f"Supplier GSTIN had formatting issues; cleaned to '{cleaned}'.",
                auto_corrected=True,
                correction_applied=f"Stripped non-alphanumeric chars → {cleaned}",
            ))
            supplier_gstin = cleaned
        else:
            errors.append(ValidationError(
                code="GST-001",
                severity=ValidationSeverity.ERROR,
                field="supplier.gstin",
                expected="SS AAAAA 9999 A Z C (15 chars)",
                found=supplier_gstin,
                message=f"Supplier GSTIN '{supplier_gstin}' fails structural format check.",
            ))

    # GST-002: Checksum validation
    if supplier_gstin and len(supplier_gstin) == 15:
        if not validate_gstin_checksum(supplier_gstin):
            errors.append(ValidationError(
                code="GST-002",
                severity=ValidationSeverity.ERROR,
                field="supplier.gstin",
                expected="valid checksum (Govt modulus-36)",
                found=supplier_gstin,
                message=(
                    f"Supplier GSTIN '{supplier_gstin}' fails checksum validation. "
                    "Likely OCR digit substitution error."
                ),
            ))

    # GST-003: State code consistency
    if supplier_gstin and len(supplier_gstin) == 15:
        gstin_sc = supplier_gstin[:2]
        stored_sc = inv.supplier.state_code
        if stored_sc and stored_sc != gstin_sc:
            errors.append(ValidationError(
                code="GST-003",
                severity=ValidationSeverity.WARNING,
                field="supplier.state_code",
                expected=gstin_sc,
                found=stored_sc,
                message=(
                    f"Supplier state code '{stored_sc}' doesn't match GSTIN "
                    f"prefix '{gstin_sc}'. Auto-correcting."
                ),
                auto_corrected=True,
                correction_applied=f"Set supplier.state_code = '{gstin_sc}'",
            ))
            corrected.supplier.state_code = gstin_sc

    # GST-004: CGST+SGST vs IGST consistency with transaction type
    for li in inv.line_items:
        if inv.transaction_type == TransactionType.INTRA_STATE:
            if li.tax.igst_amount > 0:
                errors.append(ValidationError(
                    code="GST-004",
                    severity=ValidationSeverity.ERROR,
                    field=f"line_items[{li.sl_no}].tax.igst_amount",
                    expected="0.00 (intra-state = no IGST)",
                    found=str(li.tax.igst_amount),
                    message=(
                        f"Line {li.sl_no}: IGST charged on what appears to be "
                        "an intra-state transaction."
                    ),
                ))
        elif inv.transaction_type == TransactionType.INTER_STATE:
            if li.tax.cgst_amount > 0 or li.tax.sgst_amount > 0:
                errors.append(ValidationError(
                    code="GST-004",
                    severity=ValidationSeverity.ERROR,
                    field=f"line_items[{li.sl_no}].tax.cgst_amount",
                    expected="0.00 (inter-state = no CGST/SGST)",
                    found=f"CGST={li.tax.cgst_amount}, SGST={li.tax.sgst_amount}",
                    message=(
                        f"Line {li.sl_no}: CGST/SGST charged on inter-state transaction."
                    ),
                ))

    # GST-005: Valid GST rate
    for li in inv.line_items:
        if li.tax.gst_rate not in VALID_GST_RATES:
            errors.append(ValidationError(
                code="GST-005",
                severity=ValidationSeverity.WARNING,
                field=f"line_items[{li.sl_no}].tax.gst_rate",
                expected=f"one of {sorted(VALID_GST_RATES)}",
                found=str(li.tax.gst_rate),
                message=(
                    f"Line {li.sl_no}: GST rate {li.tax.gst_rate}% is not in the "
                    "standard rate schedule."
                ),
            ))

    # GST-006: CGST = SGST for intra-state
    if inv.transaction_type == TransactionType.INTRA_STATE:
        for li in inv.line_items:
            if abs(li.tax.cgst_rate - li.tax.sgst_rate) > 0.01:
                errors.append(ValidationError(
                    code="GST-006",
                    severity=ValidationSeverity.ERROR,
                    field=f"line_items[{li.sl_no}].tax",
                    expected="cgst_rate == sgst_rate",
                    found=f"CGST={li.tax.cgst_rate}%, SGST={li.tax.sgst_rate}%",
                    message=(
                        f"Line {li.sl_no}: For intra-state, CGST rate must equal SGST rate."
                    ),
                ))

    return errors, corrected


def _check_math(inv: Invoice) -> tuple[list[ValidationError], Invoice]:
    """
    Perform all arithmetic validation checks with auto-correction.
    Returns errors and a corrected copy of the invoice.
    """
    errors: list[ValidationError] = []
    corrected = copy.deepcopy(inv)
    tol = cfg.TAX_ROUNDING_TOLERANCE

    for idx, li in enumerate(inv.line_items):
        # MATH-001: Taxable amount = qty × price − discount
        expected_taxable = round(li.quantity * li.unit_price, 2)
        if li.discount_pct:
            expected_taxable -= round(expected_taxable * li.discount_pct / 100, 2)
        elif li.discount_amount:
            expected_taxable -= li.discount_amount
        expected_taxable = round(expected_taxable, 2)

        taxable_diff = abs(li.taxable_amount - expected_taxable)
        if taxable_diff > tol:
            if taxable_diff <= 2.0:
                # Auto-correct small discrepancy
                corrected.line_items[idx].taxable_amount = expected_taxable
                corrected.line_items[idx].tax = TaxBreakdown(
                    gst_rate=li.tax.gst_rate,
                    taxable_amount=expected_taxable,
                    cgst_rate=li.tax.cgst_rate,
                    cgst_amount=round(expected_taxable * li.tax.cgst_rate / 100, 2),
                    sgst_rate=li.tax.sgst_rate,
                    sgst_amount=round(expected_taxable * li.tax.sgst_rate / 100, 2),
                    igst_rate=li.tax.igst_rate,
                    igst_amount=round(expected_taxable * li.tax.igst_rate / 100, 2),
                )
                errors.append(ValidationError(
                    code="MATH-001",
                    severity=ValidationSeverity.WARNING,
                    field=f"line_items[{li.sl_no}].taxable_amount",
                    expected=str(expected_taxable),
                    found=str(li.taxable_amount),
                    message=(
                        f"Line {li.sl_no}: Taxable amount ₹{li.taxable_amount} differs "
                        f"from computed ₹{expected_taxable} by ₹{taxable_diff:.2f}. "
                        "Auto-corrected."
                    ),
                    auto_corrected=True,
                    correction_applied=f"Set taxable_amount = {expected_taxable}",
                ))
            else:
                errors.append(ValidationError(
                    code="MATH-001",
                    severity=ValidationSeverity.ERROR,
                    field=f"line_items[{li.sl_no}].taxable_amount",
                    expected=str(expected_taxable),
                    found=str(li.taxable_amount),
                    message=(
                        f"Line {li.sl_no}: Taxable amount ₹{li.taxable_amount} differs from "
                        f"computed ₹{expected_taxable} by ₹{taxable_diff:.2f} — exceeds "
                        f"correction threshold."
                    ),
                ))

        # Use the potentially corrected line for tax checks
        check_li = corrected.line_items[idx]

        # MATH-002: CGST
        if check_li.tax.cgst_rate > 0:
            expected_cgst = round(check_li.taxable_amount * check_li.tax.cgst_rate / 100, 2)
            diff = abs(check_li.tax.cgst_amount - expected_cgst)
            if diff > tol:
                corrected.line_items[idx].tax = TaxBreakdown(
                    gst_rate=check_li.tax.gst_rate,
                    taxable_amount=check_li.taxable_amount,
                    cgst_rate=check_li.tax.cgst_rate,
                    cgst_amount=expected_cgst,
                    sgst_rate=check_li.tax.sgst_rate,
                    sgst_amount=check_li.tax.sgst_amount,
                    igst_rate=check_li.tax.igst_rate,
                    igst_amount=check_li.tax.igst_amount,
                )
                errors.append(ValidationError(
                    code="MATH-002",
                    severity=ValidationSeverity.WARNING,
                    field=f"line_items[{li.sl_no}].tax.cgst_amount",
                    expected=str(expected_cgst),
                    found=str(check_li.tax.cgst_amount),
                    message=(
                        f"Line {li.sl_no}: CGST ₹{check_li.tax.cgst_amount} ≠ "
                        f"computed ₹{expected_cgst}. Auto-corrected."
                    ),
                    auto_corrected=True,
                    correction_applied=f"Set cgst_amount = {expected_cgst}",
                ))

        # MATH-003: SGST
        if check_li.tax.sgst_rate > 0:
            expected_sgst = round(check_li.taxable_amount * check_li.tax.sgst_rate / 100, 2)
            diff = abs(check_li.tax.sgst_amount - expected_sgst)
            if diff > tol:
                corrected.line_items[idx].tax.sgst_amount = expected_sgst
                errors.append(ValidationError(
                    code="MATH-003",
                    severity=ValidationSeverity.WARNING,
                    field=f"line_items[{li.sl_no}].tax.sgst_amount",
                    expected=str(expected_sgst),
                    found=str(check_li.tax.sgst_amount),
                    message=(
                        f"Line {li.sl_no}: SGST ₹{check_li.tax.sgst_amount} ≠ "
                        f"computed ₹{expected_sgst}. Auto-corrected."
                    ),
                    auto_corrected=True,
                    correction_applied=f"Set sgst_amount = {expected_sgst}",
                ))

        # MATH-004: IGST
        if check_li.tax.igst_rate > 0:
            expected_igst = round(check_li.taxable_amount * check_li.tax.igst_rate / 100, 2)
            diff = abs(check_li.tax.igst_amount - expected_igst)
            if diff > tol:
                corrected.line_items[idx].tax.igst_amount = expected_igst
                errors.append(ValidationError(
                    code="MATH-004",
                    severity=ValidationSeverity.WARNING,
                    field=f"line_items[{li.sl_no}].tax.igst_amount",
                    expected=str(expected_igst),
                    found=str(check_li.tax.igst_amount),
                    message=(
                        f"Line {li.sl_no}: IGST ₹{check_li.tax.igst_amount} ≠ "
                        f"computed ₹{expected_igst}. Auto-corrected."
                    ),
                    auto_corrected=True,
                    correction_applied=f"Set igst_amount = {expected_igst}",
                ))

    # Recompute invoice-level totals from (possibly corrected) line items
    corrected.recompute_totals()

    # MATH-005: Subtotal
    sub_diff = abs(inv.totals.subtotal - corrected.totals.computed_subtotal)
    if sub_diff > tol:
        errors.append(ValidationError(
            code="MATH-005",
            severity=ValidationSeverity.WARNING if sub_diff <= cfg.GRAND_TOTAL_TOLERANCE else ValidationSeverity.ERROR,
            field="totals.subtotal",
            expected=str(corrected.totals.computed_subtotal),
            found=str(inv.totals.subtotal),
            message=(
                f"Invoice subtotal ₹{inv.totals.subtotal} ≠ "
                f"sum of taxable amounts ₹{corrected.totals.computed_subtotal} "
                f"(diff ₹{sub_diff:.2f})."
            ),
            auto_corrected=True,
            correction_applied=f"Set subtotal = {corrected.totals.computed_subtotal}",
        ))
        corrected.totals.subtotal = corrected.totals.computed_subtotal

    # MATH-006: Total tax
    tax_diff = abs(inv.totals.total_tax - corrected.totals.computed_total_tax)
    if tax_diff > tol:
        errors.append(ValidationError(
            code="MATH-006",
            severity=ValidationSeverity.WARNING,
            field="totals.total_tax",
            expected=str(corrected.totals.computed_total_tax),
            found=str(inv.totals.total_tax),
            message=(
                f"Invoice total tax ₹{inv.totals.total_tax} ≠ "
                f"sum of line taxes ₹{corrected.totals.computed_total_tax}. Auto-corrected."
            ),
            auto_corrected=True,
            correction_applied=f"Set total_tax = {corrected.totals.computed_total_tax}",
        ))
        corrected.totals.total_tax = corrected.totals.computed_total_tax

    # MATH-007: Grand total
    expected_grand = round(corrected.totals.subtotal + corrected.totals.total_tax + corrected.totals.round_off, 2)
    grand_diff = abs(inv.totals.grand_total - expected_grand)
    if grand_diff > cfg.GRAND_TOTAL_TOLERANCE:
        errors.append(ValidationError(
            code="MATH-007",
            severity=ValidationSeverity.ERROR,
            field="totals.grand_total",
            expected=str(expected_grand),
            found=str(inv.totals.grand_total),
            message=(
                f"Grand total ₹{inv.totals.grand_total} ≠ "
                f"subtotal + tax + round_off = ₹{expected_grand} "
                f"(diff ₹{grand_diff:.2f} exceeds tolerance ₹{cfg.GRAND_TOTAL_TOLERANCE})."
            ),
        ))
    elif grand_diff > tol:
        corrected.totals.grand_total = expected_grand
        errors.append(ValidationError(
            code="MATH-007",
            severity=ValidationSeverity.WARNING,
            field="totals.grand_total",
            expected=str(expected_grand),
            found=str(inv.totals.grand_total),
            message=(
                f"Grand total ₹{inv.totals.grand_total} ≠ ₹{expected_grand} "
                f"(diff ₹{grand_diff:.2f} within tolerance — auto-corrected)."
            ),
            auto_corrected=True,
            correction_applied=f"Set grand_total = {expected_grand}",
        ))

    return errors, corrected


def _check_masters(
    inv: Invoice,
    skus: list[SKUMaster],
    pos: list[PurchaseOrder],
) -> tuple[list[ValidationError], Invoice]:
    """
    Fuzzy-match line items against SKU master and check PO rates.
    Returns errors and invoice with match metadata populated.
    """
    errors: list[ValidationError] = []
    corrected = copy.deepcopy(inv)

    # Build fuzzy match corpus: list of (name, sku) tuples
    sku_corpus: list[tuple[str, SKUMaster]] = []
    for sku in skus:
        for name in sku.all_names:
            sku_corpus.append((name.lower(), sku))

    corpus_strings = [item[0] for item in sku_corpus]

    # Find matching PO for rate check
    matching_po = next(
        (p for p in pos if p.po_number == inv.po_number and p.is_open),
        None,
    )
    po_rate_map: dict[str, float] = {}
    if matching_po:
        for po_item in matching_po.items:
            po_rate_map[po_item["sku_code"]] = po_item["unit_price"]

    for idx, li in enumerate(inv.line_items):
        query = li.description.lower().strip()

        # SKU-001: Fuzzy match
        matches = process.extract(
            query,
            corpus_strings,
            scorer=fuzz.token_sort_ratio,
            limit=3,
        )

        best_match_name, best_score, best_corpus_idx = matches[0] if matches else ("", 0, 0)

        if best_score >= cfg.SKU_MATCH_MIN_SCORE:
            matched_sku = sku_corpus[best_corpus_idx][1]
            corrected.line_items[idx].matched_sku_code = matched_sku.sku_code
            corrected.line_items[idx].matched_sku_name = matched_sku.name
            corrected.line_items[idx].sku_match_score = best_score

            # SKU-002: HSN code cross-check
            if li.hsn_sac_code and matched_sku.hsn_sac_code:
                if li.hsn_sac_code != matched_sku.hsn_sac_code:
                    errors.append(ValidationError(
                        code="SKU-002",
                        severity=ValidationSeverity.WARNING,
                        field=f"line_items[{li.sl_no}].hsn_sac_code",
                        expected=matched_sku.hsn_sac_code,
                        found=li.hsn_sac_code,
                        message=(
                            f"Line {li.sl_no} '{li.description[:30]}': "
                            f"HSN {li.hsn_sac_code!r} ≠ master HSN {matched_sku.hsn_sac_code!r} "
                            f"(matched SKU: {matched_sku.name})."
                        ),
                    ))

            # PO-002: upward rate deviation = BLOCKING overbilling; downward = warning only
            if matched_sku.sku_code in po_rate_map:
                po_rate, billed = po_rate_map[matched_sku.sku_code], li.unit_price
                dev = (billed - po_rate) / po_rate * 100 if po_rate else 0.0  # signed %
                corrected.line_items[idx].po_rate_deviation_pct = round(dev, 2)
                over = bool(po_rate) and billed > po_rate * 1.005
                under = bool(po_rate) and billed < po_rate and abs(dev) > cfg.PO_RATE_DEVIATION_PCT
                if over or under:
                    errors.append(ValidationError(
                        code="PO-002",
                        severity=ValidationSeverity.ERROR if over else ValidationSeverity.WARNING,
                        field=f"line_items[{li.sl_no}].unit_price",
                        expected=f"<= ₹{po_rate * 1.005:.2f} (PO rate ₹{po_rate:.2f} + 0.5%)" if over else f"≈ ₹{po_rate:.2f}",
                        found=f"₹{billed:.2f}",
                        message=(f"Line {li.sl_no} '{li.description[:30]}': billed ₹{billed:.2f} is {dev:+.2f}% vs PO rate "
                                 f"₹{po_rate:.2f}. " + ("OVERBILLING — push blocked until overridden with an audit reason."
                                 if over else "Billed below PO rate (non-blocking).")),
                    ))
        else:
            errors.append(ValidationError(
                code="SKU-001",
                severity=ValidationSeverity.WARNING,
                field=f"line_items[{li.sl_no}].description",
                expected=f"fuzzy score >= {cfg.SKU_MATCH_MIN_SCORE}",
                found=f"best score = {best_score:.1f} ('{best_match_name}')",
                message=(
                    f"Line {li.sl_no} '{li.description[:40]}': "
                    f"Cannot confidently match to SKU master (best score {best_score:.1f})."
                ),
            ))

    # PO-001: PO number exists in open PO master
    if inv.po_number:
        po_found = any(p.po_number == inv.po_number for p in pos)
        if not po_found:
            errors.append(ValidationError(
                code="PO-001",
                severity=ValidationSeverity.WARNING,
                field="po_number",
                expected="a valid open PO number",
                found=inv.po_number,
                message=(
                    f"PO number '{inv.po_number}' not found in open PO master. "
                    "Walk-in invoice or unregistered PO."
                ),
            ))

    return errors, corrected


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def validate_invoice(
    raw: RawExtractionResult,
    skus: list[SKUMaster],
    pos: list[PurchaseOrder],
) -> ValidationResult:
    """
    Full validation pipeline:
      1. Convert RawExtractionResult → typed Invoice
      2. Run structural checks
      3. Run GST / GSTIN checks
      4. Run arithmetic checks (with auto-correction)
      5. Run master reconciliation checks
      6. Determine overall is_valid flag

    Args:
        raw:   The raw extraction output from Docling / VLM.
        skus:  SKU master list for fuzzy matching.
        pos:   Purchase Order master for rate-deviation checks.

    Returns:
        ValidationResult with structured errors and corrected invoice.
    """
    inv = _raw_to_invoice(raw)
    all_errors: list[ValidationError] = []

    # Pass 1: Structural
    struct_errors = _check_struct(inv)
    all_errors.extend(struct_errors)

    # Pass 2: GST checks
    gst_errors, inv = _check_gst(inv)
    all_errors.extend(gst_errors)

    # Pass 3: Math checks (operates on GST-corrected invoice)
    math_errors, inv = _check_math(inv)
    all_errors.extend(math_errors)

    # Pass 4: Master reconciliation (operates on math-corrected invoice)
    master_errors, inv = _check_masters(inv, skus, pos)
    all_errors.extend(master_errors)

    # Determine overall validity:
    # Invoice is valid if there are no blocking errors
    # (auto-corrected errors do NOT block; uncorrected ERRORs DO block)
    blocking = [e for e in all_errors if e.blocks_push]
    is_valid = len(blocking) == 0

    return ValidationResult(
        invoice_id=inv.invoice_id,
        is_valid=is_valid,
        errors=all_errors,
        corrected_invoice=inv,
    )


def validate_from_demo(
    skus: list[SKUMaster],
    pos: list[PurchaseOrder],
    seed: int = 42,
) -> tuple[RawExtractionResult, ValidationResult]:
    """
    Convenience function: generate a mock invoice and validate it.
    Returns (raw_extraction, validation_result).
    Used by app.py when running in DEMO_MODE.
    """
    from mock_data import generate_mock_invoice, PARTY_MASTER

    raw = generate_mock_invoice(PARTY_MASTER, skus, pos, seed=seed)
    result = validate_invoice(raw, skus, pos)
    return raw, result


if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table
    from mock_data import SKU_MASTER, PO_MASTER

    console = Console()

    raw, result = validate_from_demo(SKU_MASTER, PO_MASTER, seed=42)

    console.print(f"\n[bold]Invoice:[/bold] {raw.invoice_number}")
    console.print(f"[bold]Supplier:[/bold] {raw.supplier_name} ({raw.supplier_gstin})")
    console.print(f"[bold]Grand Total:[/bold] ₹{raw.grand_total}")
    console.print(f"\n[bold cyan]Validation Result:[/bold cyan] {result.summary}")

    if result.errors:
        t = Table("Code", "Severity", "Field", "Message", "Auto-Fixed?")
        for e in result.errors:
            sev_color = "red" if e.severity.value == "error" else "yellow" if e.severity.value == "warning" else "blue"
            t.add_row(
                e.code,
                f"[{sev_color}]{e.severity.value.upper()}[/{sev_color}]",
                e.field[:40],
                e.message[:60],
                "✅" if e.auto_corrected else "❌",
            )
        console.print(t)

    if result.corrected_invoice:
        inv = result.corrected_invoice
        console.print(f"\n[green]Corrected Grand Total:[/green] ₹{inv.totals.grand_total}")
        console.print(f"[green]Line Items:[/green] {inv.line_item_count}")
        for li in inv.line_items[:3]:
            console.print(
                f"  [{li.sl_no}] {li.description[:35]:<35} "
                f"QTY={li.quantity:>7.2f}  ₹{li.taxable_amount:>10.2f}  "
                f"SKU={li.matched_sku_code or 'UNMATCHED'} ({li.sku_match_score:.0f}%)"
            )
