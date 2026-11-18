"""
router_extractor.py — Hybrid document router + DoubleTake row re-verification.

Routing
───────
  DEMO_MODE + demo_seed     → synthetic invoice (mock_data) rendered to a real image,
                              with injected OCR / math / cropped-row errors
  Digital PDF               → Docling (tables + markdown)  → PyMuPDF/pdfplumber text-layer fallback
  Scanned PDF / image       → OpenCV deskew + CLAHE → VLM (Gemini/OpenAI/Anthropic) when
                              DEMO_MODE=False and a key is set, else Tesseract OCR

DoubleTake loop (runs on every route)
─────────────────────────────────────
  1. Missing-row check : stated subtotal − Σ line taxables > tolerance → re-crop the page sliver
                         under the last row (with DOUBLETAKE_OVERLAP_PX overlap), re-read, merge rows.
  2. Row-math check    : qty × rate − discount ≠ taxable (> ₹0.50) → zoomed row crop saved to
                         ./temp_crops/, re-read; the new value is accepted ONLY if it passes the math.
  Every crop is saved to ./temp_crops/ for the Streamlit side-by-side review.

Public API
──────────
  extract_invoice(source, demo_seed=None) -> ExtractionOutput
  preprocess_image(img)                   -> PreprocessResult      (deskew + CLAHE)
  render_demo_invoice_image(raw, truth)   -> np.ndarray (BGR)
"""

from __future__ import annotations

import base64
import json
import math
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger
from PIL import Image, ImageDraw, ImageFont

from config import cfg, INDIAN_STATE_CODES, VALID_GST_RATES
from schemas import (
    BoundingBox,
    DocumentType,
    ExtractionStatus,
    RawExtractionResult,
    RawLineItem,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

TEMP_CROPS_DIR = Path(__file__).resolve().parent / "temp_crops"      # == ./temp_crops/
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MAX_PAGES = 6                    # RAM guard for 16 GB laptops
MAX_SIDE_PX = 2200               # longest image side after resize
MAX_DESKEW_DEG = 15.0
CROP_ZOOM = 2.5
ROW_MATH_TOL = 0.50              # ₹ — stricter than validator so the DoubleTake catches ₹1 slips
USE_DOCLING = os.getenv("MSME_USE_DOCLING", "1").lower() not in ("0", "false", "no")

_GSTIN_STRICT = re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b")
_GSTIN_LABELLED = re.compile(r"GSTIN(?:\s*/\s*UIN)?\s*(?:No\.?)?\s*[:\-]?\s*([0-9A-Z]{15})", re.I)
_AMT = r"([\d,]+(?:\.\d{1,2})?)"
_CUR = r"(?:₹|Rs\.?|INR)?\s*"


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Word:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    page: int = 1


@dataclass
class PreprocessResult:
    deskewed: np.ndarray
    enhanced: np.ndarray
    angle_deg: float


@dataclass
class DoubleTakeEvent:
    kind: str                        # "row_math" | "missing_row"
    row_no: int
    page: int
    reason: str
    status: str                      # "recovered" | "unresolved" | "flagged"
    crop_path: str = ""              # zoomed crop (./temp_crops/…)
    orig_path: str = ""              # un-zoomed context crop
    region: dict[str, float] = field(default_factory=dict)   # normalised x,y,w,h on the page
    before: dict[str, str] = field(default_factory=dict)
    after: dict[str, str] = field(default_factory=dict)
    reader: str = ""


@dataclass
class ExtractionOutput:
    raw: RawExtractionResult
    page_images: list[np.ndarray]    # BGR pages used for the UI overlay
    events: list[DoubleTakeEvent]
    source_path: str
    route: str
    warnings: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────

def _f(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        return default
    cleaned = re.sub(r"[₹,\s%]|Rs\.?|INR", "", str(value))
    if cleaned in ("", "-", "NA", "N/A", "nil", "Nil"):
        return default
    try:
        return float(cleaned)
    except ValueError:
        return default


def _fmt(x: float) -> str:
    return f"{x:.2f}"


def _s(v: Any) -> str:
    return "" if v is None else str(v).strip()


def _safe_stem(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(name).stem)[:40] or "invoice"


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _bbox_dict(x: float, y: float, w: float, h: float, page: int = 1, conf: float = 0.9) -> dict[str, Any]:
    x, y = _clamp01(x), _clamp01(y)
    return {
        "x": round(x, 4), "y": round(y, 4),
        "w": round(min(_clamp01(w), 1.0 - x), 4), "h": round(min(_clamp01(h), 1.0 - y), 4),
        "page": page, "confidence": round(_clamp01(conf), 3),
    }


def _imencode_jpeg(img: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def _write_png(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", img)
    if ok:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(buf.tobytes())            # unicode-safe on Windows


# ─────────────────────────────────────────────────────────────────────────────
# OpenCV preprocessing: deskew + CLAHE
# ─────────────────────────────────────────────────────────────────────────────

def _limit_size(img: np.ndarray, max_side: int = MAX_SIDE_PX) -> np.ndarray:
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img
    scale = max_side / longest
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def estimate_skew_angle(gray: np.ndarray) -> float:
    """
    Estimate page skew in degrees (positive = text baseline descends to the right).
    Primary: median angle of long near-horizontal Hough lines (invoice tables).
    Fallback: ink-pixel minAreaRect.
    """
    h, w = gray.shape[:2]
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 720, threshold=max(80, w // 12),
        minLineLength=max(60, w // 4), maxLineGap=max(5, w // 50),
    )
    angles: list[float] = []
    if lines is not None:
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):     # (N,1,4) on cv2 4.x, (N,4) on 5.x
            dx, dy = x2 - x1, y2 - y1
            if abs(dx) < 1:
                continue
            a = math.degrees(math.atan2(dy, dx))
            if abs(a) <= MAX_DESKEW_DEG:
                angles.append(a)
    if len(angles) >= 3:
        return float(np.median(angles))

    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    ys, xs = np.where(thr > 0)
    if len(xs) < 500:
        return 0.0
    pts = np.column_stack([xs, ys]).astype(np.float32)
    angle = cv2.minAreaRect(pts)[-1]
    if angle > 45:
        angle -= 90
    elif angle < -45:
        angle += 90
    return float(angle) if abs(angle) <= MAX_DESKEW_DEG else 0.0


def deskew(img_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    angle = estimate_skew_angle(gray)
    if abs(angle) < 0.3:
        return img_bgr, 0.0
    h, w = img_bgr.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(
        img_bgr, m, (w, h), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
    )
    return rotated, angle


def apply_clahe(img_bgr: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation on the L channel."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_ch = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(l_ch)
    return cv2.cvtColor(cv2.merge((l_ch, a_ch, b_ch)), cv2.COLOR_LAB2BGR)


def preprocess_image(img_bgr: np.ndarray) -> PreprocessResult:
    img = _limit_size(img_bgr)
    straight, angle = deskew(img)
    return PreprocessResult(deskewed=straight, enhanced=apply_clahe(straight), angle_deg=angle)


# ─────────────────────────────────────────────────────────────────────────────
# Loading / document-type detection
# ─────────────────────────────────────────────────────────────────────────────

def _import_fitz():
    try:
        import pymupdf as fitz  # type: ignore
        return fitz
    except ImportError:
        try:
            import fitz  # type: ignore
            return fitz
        except ImportError:
            return None


def _read_image_file(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot decode image: {path.name}")
    return img


def rasterize_pdf(path: Path, dpi: int = 130, max_pages: int = MAX_PAGES) -> list[np.ndarray]:
    fitz = _import_fitz()
    if fitz is None:
        raise RuntimeError("PyMuPDF is required to rasterise PDFs (pip install pymupdf).")
    pages: list[np.ndarray] = []
    with fitz.open(str(path)) as doc:
        for idx, page in enumerate(doc):
            if idx >= max_pages:
                break
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 1:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            else:
                arr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)
            pages.append(_limit_size(arr.copy()))
    return pages


def detect_document_type(path: Path) -> DocumentType:
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return DocumentType.IMAGE
    if suffix != ".pdf":
        return DocumentType.UNKNOWN
    fitz = _import_fitz()
    try:
        if fitz is not None:
            with fitz.open(str(path)) as doc:
                chars = sum(len(p.get_text("text").strip()) for p in list(doc)[:2])
        else:
            import pdfplumber
            with pdfplumber.open(str(path)) as pdf:
                chars = sum(len((p.extract_text() or "").strip()) for p in pdf.pages[:2])
        return DocumentType.DIGITAL_PDF if chars >= 80 else DocumentType.SCANNED_PDF
    except Exception as exc:
        logger.warning(f"Document-type probe failed ({exc}); treating as scanned.")
        return DocumentType.SCANNED_PDF


# ─────────────────────────────────────────────────────────────────────────────
# Word geometry → bounding boxes
# ─────────────────────────────────────────────────────────────────────────────

def _norm_tok(t: str) -> str:
    return re.sub(r"[^\w.]", "", t).lower()


def _union(words: list[Word]) -> tuple[float, float, float, float, int]:
    x0, y0 = min(w.x0 for w in words), min(w.y0 for w in words)
    x1, y1 = max(w.x1 for w in words), max(w.y1 for w in words)
    return x0, y0, x1 - x0, y1 - y0, words[0].page


def _locate(words: list[Word], value: str) -> list[Word] | None:
    """Find the consecutive word run matching `value` (token-wise, punctuation-insensitive)."""
    tokens = [_norm_tok(t) for t in value.split() if _norm_tok(t)]
    if not tokens or not words:
        return None
    n = len(tokens)
    normed = [_norm_tok(w.text) for w in words]
    for i in range(len(words) - n + 1):
        if normed[i:i + n] == tokens and len({w.page for w in words[i:i + n]}) == 1:
            return words[i:i + n]
    longest = max(tokens, key=len)           # fall back to the longest single token
    if len(longest) >= 4:
        for i, tok in enumerate(normed):
            if tok == longest:
                return [words[i]]
    return None


def _attach_bboxes(raw: RawExtractionResult, words: list[Word]) -> None:
    """Populate raw.bboxes for header fields, line-item rows and totals from word geometry."""
    if not words:
        return
    boxes: dict[str, dict[str, Any]] = {}

    def put(key: str, value: str, pad: float = 0.004) -> None:
        if not value:
            return
        hit = _locate(words, value)
        if hit:
            x, y, w, h, pg = _union(hit)
            boxes[key] = _bbox_dict(x - pad, y - pad / 2, w + 2 * pad, h + pad, pg, 0.9)

    put("supplier_name", raw.supplier_name)
    put("supplier_gstin", raw.supplier_gstin)
    put("invoice_number", raw.invoice_number)
    put("invoice_date", raw.invoice_date)
    put("po_number", raw.po_number)
    put("buyer_name", raw.buyer_name)
    put("buyer_gstin", raw.buyer_gstin)
    put("subtotal", raw.subtotal)
    put("total_cgst", raw.total_cgst)
    put("total_sgst", raw.total_sgst)
    put("total_igst", raw.total_igst)
    put("grand_total", raw.grand_total)

    for idx, li in enumerate(raw.line_items, start=1):
        anchor = _locate(words, " ".join(li.description.split()[:3])) if li.description else None
        if not anchor and li.taxable_amount:
            anchor = _locate(words, li.taxable_amount)
        if not anchor:
            continue
        cy = (anchor[0].y0 + anchor[0].y1) / 2
        half = max((anchor[0].y1 - anchor[0].y0) * 0.7, 0.006)
        row_words = [w for w in words if w.page == anchor[0].page and abs((w.y0 + w.y1) / 2 - cy) <= half]
        _, y, _, h, pg = _union(row_words)
        boxes[f"line_item_{idx}"] = _bbox_dict(0.03, y - 0.002, 0.94, h + 0.004, pg, 0.88)

    raw.bboxes = boxes


# ─────────────────────────────────────────────────────────────────────────────
# Text → RawExtractionResult (shared by Docling / text-layer / OCR routes)
# ─────────────────────────────────────────────────────────────────────────────

def _map_header(h: str) -> str | None:
    s = re.sub(r"[^a-z0-9%]+", " ", h.lower()).strip()
    if not s:
        return None
    if "hsn" in s or "sac" in s:
        return "hsn_sac_code"
    if re.search(r"\b(sl|sr|sno|s no|no|slno)\b", s) and len(s) <= 8:
        return "sl_no"
    if any(k in s for k in ("description", "particular", "item", "product", "goods")):
        return "description"
    if "qty" in s or "quantity" in s:
        return "quantity"
    if s in ("unit", "uom", "units"):
        return "unit"
    if "disc" in s:
        return "discount_pct" if "%" in s else "discount_amount"
    for comp in ("cgst", "sgst", "igst"):
        if comp in s:
            return f"{comp}_rate" if ("%" in s or "rate" in s) else f"{comp}_amount"
    if "taxable" in s:
        return "taxable_amount"
    if ("gst" in s or "tax" in s) and ("%" in s or "rate" in s):
        return "gst_rate"
    if "gst" in s or "tax" in s:
        return "tax_amount"
    if "rate" in s or "price" in s:
        return "unit_price"
    if "total" in s or "amount" in s or "amt" in s:
        return "line_total"
    return None


def _infer_gst_rate(taxable: float, line_total: float) -> float:
    if taxable > 0 and line_total > taxable:
        r = (line_total / taxable - 1) * 100
        best = min(VALID_GST_RATES, key=lambda v: abs(v - r))
        if abs(best - r) <= 1.0:
            return best
    return 18.0


def _tax_fields(taxable: float, gst_rate: float, tax_amt: float | None, inter_state: bool) -> dict[str, str]:
    """Split a single 'tax' number into CGST/SGST/IGST fields without hiding mismatches."""
    out: dict[str, str] = {"gst_rate": _fmt(gst_rate)}
    if inter_state:
        out["igst_rate"] = _fmt(gst_rate)
        out["cgst_rate"] = out["sgst_rate"] = "0"
    else:
        out["cgst_rate"] = out["sgst_rate"] = _fmt(gst_rate / 2)
        out["igst_rate"] = "0"
    if tax_amt is None:
        return out
    full = taxable * gst_rate / 100
    slack = max(1.0, 0.01 * full)
    per_component = (not inter_state) and abs(tax_amt - full / 2) <= slack and abs(tax_amt - full) > slack
    if inter_state:
        out["igst_amount"] = _fmt(tax_amt)
    elif per_component:
        out["cgst_amount"] = out["sgst_amount"] = _fmt(tax_amt)
    else:
        out["cgst_amount"] = out["sgst_amount"] = _fmt(tax_amt / 2)
    return out


def _parse_item_line(line: str, inter_state: bool) -> RawLineItem | None:
    """Token parser for 'Sl Description… HSN Qty [Unit] Rate Taxable [GST%] [Tax] [Total]' rows."""
    tokens = line.replace("|", " ").split()
    if len(tokens) < 7 or not re.fullmatch(r"\d{1,3}", tokens[0]):
        return None
    num = re.compile(r"^[\d,]+(?:\.\d+)?%?$")
    hsn_idx = None
    for i in range(2, len(tokens) - 3):
        if re.fullmatch(r"\d{4,8}", tokens[i]) and sum(1 for t in tokens[i + 1:] if num.match(t)) >= 3:
            hsn_idx = i
            break
    if hsn_idx is None:
        return None
    desc = " ".join(tokens[1:hsn_idx])
    rest = tokens[hsn_idx + 1:]
    unit = next((t for t in rest if re.fullmatch(r"[A-Za-z]{2,5}", t)), "")
    gst_tok = next((t for t in rest if t.endswith("%") and num.match(t)), "")
    nums = [t for t in rest if num.match(t) and not t.endswith("%")]
    if len(nums) < 3:
        return None
    qty, rate, taxable = _f(nums[0]), _f(nums[1]), _f(nums[2])
    tax_amt = _f(nums[-2]) if len(nums) >= 5 else None
    line_total = _f(nums[-1]) if len(nums) >= 4 else 0.0
    gst_rate = _f(gst_tok) if gst_tok else _infer_gst_rate(taxable, line_total)
    item = RawLineItem(
        sl_no=tokens[0], description=desc, hsn_sac_code=tokens[hsn_idx],
        quantity=_fmt(qty).rstrip("0").rstrip(".") if qty != int(qty) else str(int(qty)),
        unit=unit, unit_price=_fmt(rate), taxable_amount=_fmt(taxable),
        line_total=_fmt(line_total) if line_total else "",
        **_tax_fields(taxable, gst_rate, tax_amt, inter_state),
    )
    return item


def _parse_items_from_text(text: str, inter_state: bool, start_no: int = 1) -> list[RawLineItem]:
    items: list[RawLineItem] = []
    for line in text.splitlines():
        item = _parse_item_line(line, inter_state)
        if item:
            items.append(item)
    if start_no != 1:
        for i, it in enumerate(items):
            it.sl_no = str(start_no + i)
    return items


def _rows_to_items(rows: list[list[str]], inter_state: bool) -> list[RawLineItem]:
    """Convert a table (list of rows) into RawLineItems using fuzzy header mapping."""
    header_idx, mapping = -1, {}
    for i, row in enumerate(rows[:6]):
        m = {j: _map_header(c) for j, c in enumerate(row)}
        if sum(1 for v in m.values() if v) >= 4:
            header_idx, mapping = i, {j: v for j, v in m.items() if v}
            break
    if header_idx < 0:
        return []
    items: list[RawLineItem] = []
    for row in rows[header_idx + 1:]:
        cells = {mapping[j]: _s(row[j]) for j in mapping if j < len(row)}
        first = " ".join(_s(c) for c in row[:2]).lower()
        if not any(cells.values()) or "total" in first:
            continue
        if not cells.get("description") or not (cells.get("quantity") or cells.get("taxable_amount")):
            continue
        taxable = _f(cells.get("taxable_amount"))
        qty, rate = _f(cells.get("quantity"), 1.0), _f(cells.get("unit_price"))
        total = _f(cells.get("line_total"))
        if not taxable:
            taxable = qty * rate if rate else total
        gst = _f(cells.get("gst_rate")) or _infer_gst_rate(taxable, total)
        tax_single = cells.get("tax_amount")
        d: dict[str, str] = dict(
            sl_no=cells.get("sl_no") or str(len(items) + 1),
            description=cells["description"], hsn_sac_code=cells.get("hsn_sac_code", ""),
            quantity=cells.get("quantity", "1"), unit=cells.get("unit", ""),
            unit_price=cells.get("unit_price", ""), discount_pct=cells.get("discount_pct", "0") or "0",
            discount_amount=cells.get("discount_amount", "0") or "0",
            taxable_amount=_fmt(taxable), line_total=cells.get("line_total", ""),
        )
        explicit = {k: cells[k] for k in
                    ("cgst_rate", "cgst_amount", "sgst_rate", "sgst_amount", "igst_rate", "igst_amount") if cells.get(k)}
        if explicit:
            d.update(explicit)
            d["gst_rate"] = _fmt(gst)
        else:
            d.update(_tax_fields(taxable, gst, _f(tax_single) if tax_single else None, inter_state))
        items.append(RawLineItem(**d))
    return items


_TO_DIGIT = str.maketrans("OIlSBZ", "011582")
_TO_ALPHA = str.maketrans("01582", "OISBZ")


def _fix_gstin_ocr(g: str) -> str:
    """Positional OCR repair: digit slots (0-1, 7-10) get O→0 etc., letter slots (2-6, 11) the reverse."""
    g = re.sub(r"[^0-9A-Za-z]", "", g).upper()
    if len(g) != 15:
        return g
    chars = list(g)
    for i in (0, 1, 7, 8, 9, 10):
        chars[i] = chars[i].translate(_TO_DIGIT)
    for i in (2, 3, 4, 5, 6, 11):
        chars[i] = chars[i].translate(_TO_ALPHA)
    return "".join(chars)


def _parse_header(text: str) -> dict[str, str]:
    flat = text.replace("|", " ")
    h: dict[str, str] = {}

    def first(pattern: str, flags: int = re.I) -> str:
        m = re.search(pattern, flat, flags)
        return m.group(1).strip(" :-\t") if m else ""

    h["invoice_number"] = first(r"(?:tax\s+)?invoice\s*(?:no\.?|number|#)\s*[:\-]?\s*([A-Z0-9][A-Z0-9/\-._]{2,})")
    h["invoice_date"] = first(
        r"(?:invoice\s*)?date\s*[:\-]?\s*(\d{4}-\d{2}-\d{2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})"
    )
    h["po_number"] = first(r"(?:\bP\.?O\.?|purchase\s*order)\s*(?:no\.?|ref\.?|number|#)?\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-/]{3,})")
    h["place_of_supply"] = first(r"place\s*of\s*supply\s*[:\-]\s*([A-Za-z &]+)")

    gstins = [_fix_gstin_ocr(m.group(1)) for m in _GSTIN_LABELLED.finditer(flat)] or _GSTIN_STRICT.findall(flat)
    company = cfg.COMPANY_GSTIN.upper()
    buyer = next((g for g in gstins if g == company), "")
    others = [g for g in gstins if g != buyer]
    h["supplier_gstin"] = others[0] if others else ""
    if not buyer and len(others) >= 2:
        buyer = others[1]
    h["buyer_gstin"] = buyer

    supplier = first(r"(?:supplier|seller|vendor|sold\s*by)\s*(?:name)?\s*[:\-]\s*(.+)")
    if not supplier:
        for line in flat.splitlines():
            s = line.strip()
            if re.search(r"\b(pvt|ltd|limited|llp|traders|industries|enterprises|solutions|mills|mfg|co)\b", s, re.I) \
                    and "bill to" not in s.lower():
                supplier = s
                break
    h["supplier_name"] = supplier
    h["buyer_name"] = first(r"(?:bill(?:ed)?\s*to|buyer|consignee|ship(?:ped)?\s*to)\s*[:\-]\s*(.+)")
    h["supplier_address"] = first(r"address\s*[:\-]\s*(.+)")
    return h


def _parse_totals(text: str) -> dict[str, str]:
    pats = {
        "subtotal": rf"(?:sub\s*-?\s*total|taxable\s*(?:value|amount)|total\s*taxable(?:\s*value)?)\s*[:\-]?\s*{_CUR}{_AMT}",
        "total_cgst": rf"(?:total\s*)?cgst(?:\s*amount)?\s*(?:@?\s*\d+(?:\.\d+)?\s*%)?\s*[:\-]?\s*{_CUR}{_AMT}\s*$",
        "total_sgst": rf"(?:total\s*)?sgst(?:\s*amount)?\s*(?:@?\s*\d+(?:\.\d+)?\s*%)?\s*[:\-]?\s*{_CUR}{_AMT}\s*$",
        "total_igst": rf"(?:total\s*)?igst(?:\s*amount)?\s*(?:@?\s*\d+(?:\.\d+)?\s*%)?\s*[:\-]?\s*{_CUR}{_AMT}\s*$",
        "total_tax": rf"total\s*(?:gst|tax)(?:\s*amount)?\s*[:\-]?\s*{_CUR}{_AMT}",
        "round_off": r"round(?:ing)?\s*off\s*[:\-]?\s*([\-+]?[\d,]+(?:\.\d{1,2})?)",
        "grand_total": rf"(?:grand\s*total|total\s*invoice\s*value|invoice\s*total|net\s*payable|total\s*amount\s*payable)\s*[:\-]?\s*{_CUR}{_AMT}",
    }
    found: dict[str, str] = {}
    for line in text.replace("|", " ").splitlines():
        s = line.strip()
        for key, pat in pats.items():
            m = re.search(pat, s, re.I)
            if m:
                found[key] = m.group(1).replace(",", "")
    return found


def _text_to_raw(
    text: str,
    tables: list[list[list[str]]],
    *,
    filename: str,
    doc_type: DocumentType,
    model_label: str,
    page_count: int,
) -> RawExtractionResult:
    hdr = _parse_header(text)
    totals = _parse_totals(text)
    sup_g, buy_g = hdr["supplier_gstin"], hdr["buyer_gstin"]
    inter_state = bool(sup_g and buy_g and sup_g[:2] != buy_g[:2])

    items: list[RawLineItem] = []
    for rows in tables:
        items = _rows_to_items(rows, inter_state)
        if items:
            break
    if not items:
        items = _parse_items_from_text(text, inter_state)

    total_tax = totals.get("total_tax") or ""
    if not total_tax and any(k in totals for k in ("total_cgst", "total_sgst", "total_igst")):
        total_tax = _fmt(sum(_f(totals.get(k)) for k in ("total_cgst", "total_sgst", "total_igst")))

    return RawExtractionResult(
        source_filename=filename, document_type=doc_type.value,
        extraction_status="success" if items else "failed",
        vlm_model_used=model_label, page_count=page_count, raw_text=text,
        invoice_number=hdr["invoice_number"], invoice_date=hdr["invoice_date"],
        po_number=hdr["po_number"], place_of_supply=hdr["place_of_supply"],
        supplier_name=hdr["supplier_name"], supplier_gstin=sup_g,
        supplier_address=hdr["supplier_address"], supplier_state_code=sup_g[:2] if sup_g else "",
        supplier_pan=sup_g[2:12] if len(sup_g) == 15 else "",
        buyer_name=hdr["buyer_name"], buyer_gstin=buy_g,
        buyer_state_code=buy_g[:2] if buy_g else cfg.COMPANY_STATE_CODE,
        line_items=items,
        subtotal=totals.get("subtotal", ""), total_cgst=totals.get("total_cgst", ""),
        total_sgst=totals.get("total_sgst", ""), total_igst=totals.get("total_igst", ""),
        total_tax=total_tax, round_off=totals.get("round_off", "0"),
        grand_total=totals.get("grand_total", ""),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Digital-PDF path: Docling → text-layer fallback
# ─────────────────────────────────────────────────────────────────────────────

_DOCLING_CONVERTER: Any = None


def _docling_extract(path: Path) -> tuple[str, list[list[list[str]]]] | None:
    """Docling markdown + tables. Returns None on ANY failure (offline model download, missing pkg…)."""
    global _DOCLING_CONVERTER
    if not USE_DOCLING:
        return None
    try:
        from docling.document_converter import DocumentConverter  # heavy import → lazy
    except Exception as exc:
        logger.info(f"Docling unavailable ({type(exc).__name__}); using text-layer fallback.")
        return None
    try:
        if _DOCLING_CONVERTER is None:
            _DOCLING_CONVERTER = DocumentConverter()
        doc = _DOCLING_CONVERTER.convert(str(path)).document
        markdown = doc.export_to_markdown()
        tables: list[list[list[str]]] = []
        for tbl in getattr(doc, "tables", []):
            try:
                df = tbl.export_to_dataframe(doc)
            except TypeError:
                df = tbl.export_to_dataframe()
            rows = [[str(c) for c in df.columns]]
            rows += [["" if v is None else str(v) for v in r] for r in df.values.tolist()]
            tables.append(rows)
        return markdown, tables
    except Exception as exc:
        logger.warning(f"Docling failed ({type(exc).__name__}: {exc}); using text-layer fallback.")
        return None


def _text_layer_extract(path: Path) -> tuple[str, list[list[list[str]]], list[Word], int]:
    """PyMuPDF text + word geometry, pdfplumber tables. Works fully offline."""
    fitz = _import_fitz()
    text_parts: list[str] = []
    words: list[Word] = []
    page_count = 1
    if fitz is not None:
        with fitz.open(str(path)) as doc:
            page_count = len(doc)
            for pno, page in enumerate(doc, start=1):
                if pno > MAX_PAGES:
                    break
                text_parts.append(page.get_text("text"))
                pw, ph = float(page.rect.width), float(page.rect.height)
                for x0, y0, x1, y1, wtxt, *_ in page.get_text("words"):
                    words.append(Word(wtxt, x0 / pw, y0 / ph, x1 / pw, y1 / ph, pno))
    tables: list[list[list[str]]] = []
    try:
        import pdfplumber
        with pdfplumber.open(str(path)) as pdf:
            page_count = max(page_count, len(pdf.pages))
            for pno, page in enumerate(pdf.pages[:MAX_PAGES], start=1):
                if fitz is None:
                    text_parts.append(page.extract_text() or "")
                    for w in page.extract_words():
                        words.append(Word(w["text"], w["x0"] / page.width, w["top"] / page.height,
                                          w["x1"] / page.width, w["bottom"] / page.height, pno))
                for t in page.extract_tables():
                    tables.append([[(c or "").strip() for c in r] for r in t])
    except Exception as exc:
        logger.info(f"pdfplumber table extraction skipped ({exc}).")
    return "\n".join(text_parts), tables, words, page_count


# ─────────────────────────────────────────────────────────────────────────────
# Tesseract OCR path
# ─────────────────────────────────────────────────────────────────────────────

def tesseract_available() -> bool:
    if cfg.TESSERACT_CMD and Path(cfg.TESSERACT_CMD).exists():
        return True
    return shutil.which("tesseract") is not None


def _ocr_page(img_bgr: np.ndarray, page_no: int, config: str | None = None) -> tuple[str, list[Word]]:
    import pytesseract
    if cfg.TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = cfg.TESSERACT_CMD
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    conf = config or cfg.TESSERACT_CONFIG
    try:
        data = pytesseract.image_to_data(rgb, config=conf, output_type=pytesseract.Output.DICT)
    except pytesseract.TesseractError:
        data = pytesseract.image_to_data(rgb, config=conf.replace("eng+hin", "eng"),
                                         output_type=pytesseract.Output.DICT)
    h, w = img_bgr.shape[:2]
    words: list[Word] = []
    lines: dict[tuple[int, int, int], list[str]] = {}
    for i, txt in enumerate(data["text"]):
        txt = (txt or "").strip()
        if not txt or float(data["conf"][i]) < 0:
            continue
        x, y, bw, bh = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        words.append(Word(txt, x / w, y / h, (x + bw) / w, (y + bh) / h, page_no))
        lines.setdefault((data["block_num"][i], data["par_num"][i], data["line_num"][i]), []).append(txt)
    return "\n".join(" ".join(v) for v in lines.values()), words


# ─────────────────────────────────────────────────────────────────────────────
# VLM path (Gemini / OpenAI / Anthropic)
# ─────────────────────────────────────────────────────────────────────────────

_INVOICE_PROMPT = """You are an expert reader of Indian GST tax invoices. Read the attached invoice page image(s) and
return ONLY one JSON object (no markdown, no commentary) with EXACTLY these keys. Use strings for every value,
copy numbers exactly as printed (no currency symbols), and use "" when a field is absent. NEVER guess digits.

{"invoice_number":"","invoice_date":"","po_number":"","place_of_supply":"",
 "supplier_name":"","supplier_gstin":"","supplier_address":"","supplier_pan":"",
 "buyer_name":"","buyer_gstin":"","buyer_address":"",
 "line_items":[{"sl_no":"","description":"","hsn_sac_code":"","quantity":"","unit":"","unit_price":"",
   "discount_pct":"0","discount_amount":"0","taxable_amount":"","gst_rate":"",
   "cgst_rate":"","cgst_amount":"","sgst_rate":"","sgst_amount":"","igst_rate":"","igst_amount":"","line_total":""}],
 "subtotal":"","total_cgst":"","total_sgst":"","total_igst":"","total_tax":"","round_off":"0",
 "grand_total":"","grand_total_in_words":"",
 "bboxes":{"supplier_gstin":{"x":0.0,"y":0.0,"w":0.0,"h":0.0,"page":1,"confidence":0.9},
           "line_item_1":{"x":0.0,"y":0.0,"w":0.0,"h":0.0,"page":1,"confidence":0.9}}}

Rules: include EVERY line item row, including rows touching the bottom edge of the page. In "bboxes" give a
normalised (0-1, origin top-left, relative to the page image) box for each of: invoice_number, invoice_date,
po_number, supplier_name, supplier_gstin, buyer_name, buyer_gstin, subtotal, total_cgst, total_sgst, total_igst,
grand_total and every line as line_item_<sl_no>."""

_ROW_PROMPT = """This image is a zoomed crop of ONE row of an Indian GST invoice line-item table. Return ONLY a JSON
object with string values: {"sl_no":"","description":"","hsn_sac_code":"","quantity":"","unit":"","unit_price":"",
"discount_pct":"0","taxable_amount":"","gst_rate":"","cgst_amount":"","sgst_amount":"","igst_amount":"","line_total":""}.
Copy digits exactly as printed. Use "" for anything not visible."""

_ROWS_PROMPT = """This image is a zoomed crop of the BOTTOM part of an invoice line-item table (the top edge may overlap
the previous row, which you must skip). Return ONLY a JSON object {"line_items":[{...}]} listing every row whose
serial number is >= {start_no} (or, if serials are missing, every row below the overlap). Each item uses string values
and keys: sl_no, description, hsn_sac_code, quantity, unit, unit_price, discount_pct, taxable_amount, gst_rate,
cgst_rate, cgst_amount, sgst_rate, sgst_amount, igst_rate, igst_amount, line_total, plus optional
"bbox":{"x":0.0,"y":0.0,"w":1.0,"h":0.1} normalised to THIS crop. Copy digits exactly as printed."""


def _call_vlm(images: list[np.ndarray], prompt: str) -> str:
    """Send images + prompt to the configured provider; returns the raw text reply."""
    from tenacity import retry, stop_after_attempt, wait_exponential

    provider = cfg.VLM_PROVIDER
    key = cfg.vlm_api_key()
    jpegs = [_imencode_jpeg(i) for i in images]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
    def _go() -> str:
        if provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=key)
            model = genai.GenerativeModel(cfg.GEMINI_MODEL)
            parts: list[Any] = [prompt] + [Image.open(__import__("io").BytesIO(b)) for b in jpegs]
            resp = model.generate_content(
                parts, generation_config={"temperature": 0, "response_mime_type": "application/json"}
            )
            return resp.text
        if provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=key)
            content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for b in jpegs:
                content.append({"type": "image_url",
                                "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(b).decode()}})
            r = client.chat.completions.create(
                model=cfg.OPENAI_MODEL, temperature=0, response_format={"type": "json_object"},
                messages=[{"role": "user", "content": content}],
            )
            return r.choices[0].message.content or ""
        if provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=key)
            blocks: list[dict[str, Any]] = [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                             "data": base64.b64encode(b).decode()}} for b in jpegs
            ]
            blocks.append({"type": "text", "text": prompt})
            r = client.messages.create(
                model=cfg.ANTHROPIC_MODEL, max_tokens=4096, temperature=0,
                messages=[{"role": "user", "content": blocks}],
            )
            return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        raise ValueError(f"Unknown VLM provider {provider!r}")

    return _go()


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("VLM reply contained no JSON object.")
    return json.loads(cleaned[start:end + 1])


_RAW_ITEM_KEYS = set(RawLineItem.model_fields.keys())


def _dict_to_item(d: dict[str, Any]) -> RawLineItem:
    return RawLineItem(**{k: _s(v) for k, v in d.items() if k in _RAW_ITEM_KEYS and not isinstance(v, (dict, list))})


def _json_to_raw(data: dict[str, Any], *, filename: str, doc_type: DocumentType, model: str, pages: int) -> RawExtractionResult:
    gs = lambda k: _s(data.get(k))  # noqa: E731
    sup_g, buy_g = gs("supplier_gstin").upper(), gs("buyer_gstin").upper()
    boxes: dict[str, dict[str, Any]] = {}
    for key, bb in (data.get("bboxes") or {}).items():
        try:
            boxes[key] = _bbox_dict(float(bb["x"]), float(bb["y"]), float(bb["w"]), float(bb["h"]),
                                    int(bb.get("page", 1)), float(bb.get("confidence", 0.8)))
        except (KeyError, TypeError, ValueError):
            continue
    items = [_dict_to_item(i) for i in (data.get("line_items") or []) if isinstance(i, dict)]
    # bbox keys are keyed by sl_no from the VLM; re-key by position for the validator
    for pos, it in enumerate(items, start=1):
        alt = boxes.pop(f"line_item_{_s(it.sl_no)}", None)
        if alt is not None and f"line_item_{pos}" not in boxes:
            boxes[f"line_item_{pos}"] = alt
    return RawExtractionResult(
        source_filename=filename, document_type=doc_type.value,
        extraction_status="success" if items else "failed", vlm_model_used=model, page_count=pages,
        raw_text=json.dumps(data, ensure_ascii=False)[:6000],
        invoice_number=gs("invoice_number"), invoice_date=gs("invoice_date"), po_number=gs("po_number"),
        place_of_supply=gs("place_of_supply"),
        supplier_name=gs("supplier_name"), supplier_gstin=sup_g, supplier_address=gs("supplier_address"),
        supplier_state_code=sup_g[:2] if sup_g else "", supplier_pan=gs("supplier_pan"),
        buyer_name=gs("buyer_name"), buyer_gstin=buy_g, buyer_address=gs("buyer_address"),
        buyer_state_code=buy_g[:2] if buy_g else cfg.COMPANY_STATE_CODE,
        line_items=items, subtotal=gs("subtotal"), total_cgst=gs("total_cgst"), total_sgst=gs("total_sgst"),
        total_igst=gs("total_igst"), total_tax=gs("total_tax"), round_off=gs("round_off") or "0",
        grand_total=gs("grand_total"), grand_total_in_words=gs("grand_total_in_words"), bboxes=boxes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Row readers used by the DoubleTake loop
# ─────────────────────────────────────────────────────────────────────────────

class RowReader:
    """Interface: re-read a single zoomed row, or a zoomed bottom sliver containing several rows."""
    name = "none"

    def read_row(self, crop: np.ndarray, row_no: int) -> RawLineItem | None:
        return None

    def read_rows(self, crop: np.ndarray, start_no: int) -> list[tuple[RawLineItem, dict[str, float] | None]]:
        return []


class DemoReader(RowReader):
    """Simulated VLM re-read: serves the 'ground truth' rows of the synthetic invoice."""
    name = "demo_mock_reread"

    def __init__(self, truth: RawExtractionResult) -> None:
        self.truth = truth

    def read_row(self, crop: np.ndarray, row_no: int) -> RawLineItem | None:
        if 1 <= row_no <= len(self.truth.line_items):
            return self.truth.line_items[row_no - 1].model_copy()
        return None

    def read_rows(self, crop: np.ndarray, start_no: int) -> list[tuple[RawLineItem, dict[str, float] | None]]:
        return [(it.model_copy(), None) for it in self.truth.line_items[start_no - 1:]]


class VLMReader(RowReader):
    name = "vlm_reread"

    def read_row(self, crop: np.ndarray, row_no: int) -> RawLineItem | None:
        try:
            data = _extract_json(_call_vlm([crop], _ROW_PROMPT))
            item = _dict_to_item(data)
            item.sl_no = str(row_no)
            return item
        except Exception as exc:
            logger.warning(f"VLM row re-read failed: {exc}")
            return None

    def read_rows(self, crop: np.ndarray, start_no: int) -> list[tuple[RawLineItem, dict[str, float] | None]]:
        try:
            data = _extract_json(_call_vlm([crop], _ROWS_PROMPT.replace("{start_no}", str(start_no))))
        except Exception as exc:
            logger.warning(f"VLM sliver re-read failed: {exc}")
            return []
        out: list[tuple[RawLineItem, dict[str, float] | None]] = []
        for d in data.get("line_items", []):
            if not isinstance(d, dict):
                continue
            bb = d.get("bbox") if isinstance(d.get("bbox"), dict) else None
            try:
                box = {k: float(bb[k]) for k in ("x", "y", "w", "h")} if bb else None
            except (KeyError, TypeError, ValueError):
                box = None
            out.append((_dict_to_item(d), box))
        return out


class OCRReader(RowReader):
    name = "tesseract_reread"

    def __init__(self, inter_state: bool) -> None:
        self.inter_state = inter_state

    def _text(self, crop: np.ndarray, psm: int) -> str:
        text, _ = _ocr_page(crop, 1, config=f"--oem 3 --psm {psm} -l eng")
        return text

    def read_row(self, crop: np.ndarray, row_no: int) -> RawLineItem | None:
        line = self._text(crop, 7)
        if not re.match(r"^\s*\d{1,3}\s", line):
            line = f"{row_no} {line}"
        item = _parse_item_line(line, self.inter_state)
        if item:
            item.sl_no = str(row_no)
        return item

    def read_rows(self, crop: np.ndarray, start_no: int) -> list[tuple[RawLineItem, dict[str, float] | None]]:
        items = _parse_items_from_text(self._text(crop, 6), self.inter_state)
        return [(it, None) for it in items if _f(it.sl_no) >= start_no or start_no == 1]


# ─────────────────────────────────────────────────────────────────────────────
# DoubleTake loop
# ─────────────────────────────────────────────────────────────────────────────

def _expected_taxable(li: RawLineItem) -> float:
    gross = _f(li.quantity) * _f(li.unit_price)
    disc = gross * _f(li.discount_pct) / 100 if _f(li.discount_pct) else _f(li.discount_amount)
    return round(gross - disc, 2)


def _row_math_ok(li: RawLineItem) -> bool:
    return abs(_f(li.taxable_amount) - _expected_taxable(li)) <= ROW_MATH_TOL


def _subtotal_gap(raw: RawExtractionResult) -> float | None:
    """Stated subtotal − Σ extracted line taxables (₹); None if no stated subtotal is available."""
    stated = _f(raw.subtotal)
    if stated <= 0:
        derived = _f(raw.grand_total) - _f(raw.total_tax) - _f(raw.round_off)
        stated = derived if derived > 0 and _f(raw.total_tax) > 0 else 0.0
    if stated <= 0:
        return None
    return round(stated - sum(_f(li.taxable_amount) for li in raw.line_items), 2)


def _crop_region(page: np.ndarray, x: float, y: float, w: float, h: float,
                 pad_x: float = 0.01, pad_y: float = 0.004) -> tuple[np.ndarray, dict[str, float]]:
    H, W = page.shape[:2]
    x0, y0 = _clamp01(x - pad_x), _clamp01(y - pad_y)
    x1, y1 = _clamp01(x + w + pad_x), _clamp01(y + h + pad_y)
    px0, py0, px1, py1 = int(x0 * W), int(y0 * H), max(int(x1 * W), int(x0 * W) + 8), max(int(y1 * H), int(y0 * H) + 8)
    crop = page[py0:py1, px0:px1]
    return crop, {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}


def _zoom(crop: np.ndarray) -> np.ndarray:
    zoomed = cv2.resize(crop, None, fx=CROP_ZOOM, fy=CROP_ZOOM, interpolation=cv2.INTER_CUBIC)
    return apply_clahe(zoomed)


def _save_crops(page: np.ndarray, region: tuple[float, float, float, float], stem: str, tag: str) -> tuple[np.ndarray, str, str, dict[str, float]]:
    crop, norm = _crop_region(page, *region)
    zoomed = _zoom(crop)
    zpath = TEMP_CROPS_DIR / f"{stem}_{tag}_zoom.png"
    opath = TEMP_CROPS_DIR / f"{stem}_{tag}_orig.png"
    _write_png(zpath, zoomed)
    _write_png(opath, crop)
    return zoomed, str(zpath), str(opath), norm


def _demo_row_bbox(k: int) -> dict[str, Any]:
    return _bbox_dict(0.03, 0.30 + (k - 1) * 0.045, 0.94, 0.045, 1, 0.9)


def _row_snapshot(li: RawLineItem) -> dict[str, str]:
    return {"description": li.description, "quantity": li.quantity, "unit_price": li.unit_price,
            "taxable_amount": li.taxable_amount}


def run_doubletake(
    raw: RawExtractionResult,
    pages: list[np.ndarray],
    reader: RowReader,
    stem: str,
    *,
    demo: bool = False,
) -> list[DoubleTakeEvent]:
    """Mutates `raw` in place; returns the list of events (each with saved ./temp_crops/ files)."""
    events: list[DoubleTakeEvent] = []
    if not pages:
        return events
    tol = cfg.GRAND_TOTAL_TOLERANCE

    # ── A. Missing-row recovery ─────────────────────────────────────────────
    for attempt in range(1, cfg.DOUBLETAKE_MAX_RETRIES + 1):
        gap = _subtotal_gap(raw)
        if gap is None or gap <= tol:
            break
        n = len(raw.line_items)
        stated = gap + sum(_f(li.taxable_amount) for li in raw.line_items)
        ratio = (stated - gap) / stated if stated else 1.0
        logger.info(f"DoubleTake[{attempt}] subtotal gap ₹{gap:.2f} (ratio {ratio:.2f}, "
                    f"trigger<{cfg.DOUBLETAKE_TRIGGER_THRESHOLD}) — re-cropping below row {n}")

        last = raw.bboxes.get(f"line_item_{n}") if n else None
        page_no = int(last["page"]) if last else len(pages)
        page = pages[min(max(page_no, 1), len(pages)) - 1]
        H = page.shape[0]
        if last:
            bottom = (last["y"] + last["h"]) * H
            row_h = max(last["h"] * H, 24.0)
        else:
            bottom, row_h = 0.55 * H, 0.045 * H
        y0 = max(0.0, bottom - cfg.DOUBLETAKE_OVERLAP_PX)
        y1 = min(float(H), bottom + 3.0 * row_h)
        foot = [raw.bboxes[k]["y"] * H for k in ("subtotal", "total_cgst", "total_sgst", "total_igst", "grand_total")
                if k in raw.bboxes]
        if foot and min(foot) > y0 + row_h:
            y1 = min(y1, min(foot))

        region = (0.0, y0 / H, 1.0, (y1 - y0) / H)
        zoomed, zpath, opath, norm = _save_crops(page, region, stem, f"missing_{attempt}")
        found = reader.read_rows(zoomed, n + 1)

        added = 0
        for item, box in found:
            dup = any(
                li.description.strip().lower() == item.description.strip().lower()
                and abs(_f(li.taxable_amount) - _f(item.taxable_amount)) < 0.01 for li in raw.line_items
            )
            if dup or not item.description.strip():
                continue
            item.sl_no = str(len(raw.line_items) + 1)
            k = len(raw.line_items) + 1
            if box:
                raw.bboxes[f"line_item_{k}"] = _bbox_dict(
                    0.03, norm["y"] + box["y"] * norm["h"], 0.94, max(box["h"] * norm["h"], 0.02), page_no, 0.8)
            elif demo:
                raw.bboxes[f"line_item_{k}"] = _demo_row_bbox(k)
            else:
                prev = raw.bboxes.get(f"line_item_{k - 1}")
                ry = (prev["y"] + prev["h"]) if prev else y0 / H
                raw.bboxes[f"line_item_{k}"] = _bbox_dict(0.03, ry, 0.94, row_h / H, page_no, 0.6)
            raw.line_items.append(item)
            added += 1

        new_gap = _subtotal_gap(raw)
        closed = new_gap is not None and new_gap <= tol
        raw.doubletake_triggered = True
        raw.doubletake_rows_recovered += added
        events.append(DoubleTakeEvent(
            kind="missing_row", row_no=n + 1, page=page_no, region={**norm},
            reason=f"Σ line taxables is ₹{gap:,.2f} short of the stated subtotal — last row likely cropped.",
            status="recovered" if (added and closed) else ("unresolved" if not added else "recovered"),
            crop_path=zpath, orig_path=opath, reader=reader.name,
            before={"rows": str(n), "subtotal_gap": _fmt(gap)},
            after={"rows": str(len(raw.line_items)), "subtotal_gap": _fmt(new_gap if new_gap is not None else 0.0)},
        ))
        if not added:
            break

    # ── B. Row-math re-verification ─────────────────────────────────────────
    for idx, li in enumerate(raw.line_items, start=1):
        if _row_math_ok(li):
            continue
        bb = raw.bboxes.get(f"line_item_{idx}")
        before = _row_snapshot(li)
        reason = (f"Row {idx}: qty × rate − discount = ₹{_expected_taxable(li):,.2f} "
                  f"but taxable reads ₹{_f(li.taxable_amount):,.2f}.")
        if not bb:
            events.append(DoubleTakeEvent("row_math", idx, 1, reason + " (no bounding box — review manually)", "flagged", before=before))
            raw.doubletake_triggered = True
            continue
        page_no = int(bb["page"])
        page = pages[min(max(page_no, 1), len(pages)) - 1]
        zoomed, zpath, opath, norm = _save_crops(page, (bb["x"], bb["y"], bb["w"], bb["h"]), stem, f"row{idx}")
        reread = reader.read_row(zoomed, idx)
        status, after = "unresolved" if reader.name != "none" else "flagged", dict(before)
        if reread is not None and _row_math_ok(reread):
            changed = False
            for fld in ("quantity", "unit_price", "discount_pct", "taxable_amount", "gst_rate",
                        "cgst_amount", "sgst_amount", "igst_amount", "line_total"):
                new_val = getattr(reread, fld, "")
                if new_val and new_val != getattr(li, fld, ""):
                    setattr(li, fld, new_val)
                    changed = True
            if changed:
                status, after = "recovered", _row_snapshot(li)
        raw.doubletake_triggered = True
        if status == "recovered":
            raw.doubletake_rows_recovered += 0      # row corrected, not added
        events.append(DoubleTakeEvent(
            "row_math", idx, page_no, reason, status, crop_path=zpath, orig_path=opath,
            region=norm, before=before, after=after, reader=reader.name,
        ))

    if raw.doubletake_triggered:
        gap = _subtotal_gap(raw)
        clean = (gap is None or gap <= tol) and all(_row_math_ok(li) for li in raw.line_items)
        raw.extraction_status = ExtractionStatus.DOUBLETAKE_RECOVERED.value if clean else ExtractionStatus.PARTIAL.value
    return events


# ─────────────────────────────────────────────────────────────────────────────
# Demo: synthetic invoice → real image
# ─────────────────────────────────────────────────────────────────────────────

def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = (["DejaVuSans-Bold.ttf", "arialbd.ttf", "LiberationSans-Bold.ttf"] if bold
             else ["DejaVuSans.ttf", "arial.ttf", "LiberationSans-Regular.ttf"])
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def render_demo_invoice_image(raw: RawExtractionResult, truth: RawExtractionResult, true_gstin: str = "",
                              width: int = 1240, height: int = 1754) -> np.ndarray:
    """
    Draw an A4 GST invoice whose text sits exactly inside the mock bounding boxes.
    Shows the TRUE values (all rows, correct GSTIN, correct taxables) — i.e. what the
    paper says — while `raw` holds what the (error-injected) extractor misread.
    """
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    f_s, f_m, f_b, f_t = _font(17), _font(19), _font(19, True), _font(26, True)
    ink = (20, 20, 20)

    def at(key: str, text: str, font: ImageFont.ImageFont = f_m) -> None:
        bb = raw.bboxes.get(key)
        if bb:
            d.text((bb["x"] * width + 4, bb["y"] * height + 3), text, fill=ink, font=font)

    d.text((width * 0.40, 4), "TAX INVOICE", fill=ink, font=f_t)
    d.rectangle([int(0.03 * width), int(0.025 * height), int(0.97 * width), int(0.275 * height)], outline=(120, 120, 120), width=2)
    at("supplier_name", truth.supplier_name or raw.supplier_name, f_b)
    at("supplier_gstin", f"GSTIN: {true_gstin or raw.supplier_gstin}")
    at("supplier_address", raw.supplier_address, f_s)
    at("invoice_number", f"Invoice No: {raw.invoice_number}")
    at("invoice_date", f"Date: {raw.invoice_date}")
    at("po_number", f"PO Ref: {raw.po_number}")
    at("buyer_name", f"Bill To: {raw.buyer_name}", f_b)
    at("buyer_gstin", f"GSTIN: {raw.buyer_gstin}")
    at("buyer_address", raw.buyer_address, f_s)

    cols = [("Sl", 0.04), ("Description", 0.07), ("HSN", 0.40), ("Qty", 0.50), ("Unit", 0.57),
            ("Rate", 0.62), ("Taxable", 0.70), ("GST%", 0.79), ("Tax", 0.85), ("Total", 0.90)]
    ty = int(0.272 * height)
    d.rectangle([int(0.03 * width), ty, int(0.97 * width), ty + 30], fill=(228, 232, 240), outline=(90, 90, 90))
    for label, cx in cols:
        d.text((cx * width, ty + 5), label, fill=ink, font=f_b)

    for i, it in enumerate(truth.line_items, start=1):
        y = int((0.30 + (i - 1) * 0.045) * height)
        d.line([(int(0.03 * width), y), (int(0.97 * width), y)], fill=(190, 190, 190), width=1)
        tax_amt = _f(it.cgst_amount) + _f(it.sgst_amount) + _f(it.igst_amount)
        cells = [it.sl_no, it.description[:34], it.hsn_sac_code, it.quantity, it.unit, it.unit_price,
                 it.taxable_amount, f"{_f(it.gst_rate):g}%", _fmt(tax_amt), it.line_total]
        for (label, cx), txt in zip(cols, cells):
            d.text((cx * width, y + 12), str(txt), fill=ink, font=f_s)

    for key, label in (("subtotal", "Subtotal"), ("total_cgst", "CGST"), ("total_sgst", "SGST"),
                       ("total_igst", "IGST"), ("grand_total", "Grand Total")):
        bb = raw.bboxes.get(key)
        if bb:
            val = _f(getattr(raw, key))
            font = f_b if key == "grand_total" else f_m
            d.text((bb["x"] * width + 4, bb["y"] * height + 3), f"{label}: {val:,.2f}", fill=ink, font=font)
    d.text((0.05 * width, 0.955 * height), raw.grand_total_in_words, fill=ink, font=f_s)
    d.text((0.05 * width, 0.975 * height), "Subject to jurisdiction. E&OE. This is a computer generated invoice.", fill=(120, 120, 120), font=f_s)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _extract_demo(seed: int) -> ExtractionOutput:
    from mock_data import PARTY_MASTER, SKU_MASTER, PO_MASTER, generate_mock_invoice

    t0 = time.perf_counter()
    raw = generate_mock_invoice(PARTY_MASTER, SKU_MASTER, PO_MASTER, seed=seed, inject_errors=True)
    truth = generate_mock_invoice(PARTY_MASTER, SKU_MASTER, PO_MASTER, seed=seed, inject_errors=False)
    true_gstin = next((p.gstin for p in PARTY_MASTER if p.name == raw.supplier_name), "")

    # The mock pre-sets DoubleTake flags; here they must come from real detection.
    raw.doubletake_triggered, raw.doubletake_rows_recovered = False, 0
    page = render_demo_invoice_image(raw, truth, true_gstin)
    events = run_doubletake(raw, [page], DemoReader(truth), _safe_stem(raw.source_filename), demo=True)
    raw.processing_time_ms = round((time.perf_counter() - t0) * 1000, 1)
    return ExtractionOutput(
        raw=raw, page_images=[page], events=events, source_path=f"demo://seed/{seed}",
        route="DEMO · synthetic invoice → simulated VLM → DoubleTake (simulated re-read)",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Router entry point
# ─────────────────────────────────────────────────────────────────────────────

def _failed(filename: str, doc_type: DocumentType, msg: str) -> RawExtractionResult:
    return RawExtractionResult(source_filename=filename, document_type=doc_type.value,
                               extraction_status="failed", raw_text=msg)


def extract_invoice(source: str | Path | None = None, *, demo_seed: int | None = None) -> ExtractionOutput:
    """
    Route a document to the right extractor and run the DoubleTake loop.

    DEMO_MODE=True + demo_seed → fully synthetic pipeline (no files, no keys).
    Otherwise `source` must be a PDF / image path.
    """
    if cfg.DEMO_MODE and demo_seed is not None:
        return _extract_demo(demo_seed)
    if source is None:
        raise ValueError("Provide a file path (or demo_seed in DEMO_MODE).")

    t0 = time.perf_counter()
    path = Path(source)
    warnings: list[str] = []
    dtype = detect_document_type(path)
    stem = _safe_stem(path.name)
    use_vlm = (not cfg.DEMO_MODE) and cfg.vlm_is_configured()
    pages: list[np.ndarray] = []
    raw: RawExtractionResult | None = None
    route = ""

    if dtype == DocumentType.UNKNOWN:
        raw = _failed(path.name, dtype, f"Unsupported file type {path.suffix!r}.")
        return ExtractionOutput(raw, [], [], str(path), "unsupported", [raw.raw_text])

    # ── Digital PDF: Docling → text layer ───────────────────────────────────
    if dtype == DocumentType.DIGITAL_PDF:
        pages = rasterize_pdf(path, dpi=130)
        text_layer, tl_tables, words, page_count = _text_layer_extract(path)
        doc = _docling_extract(path)
        if doc is not None:
            md, d_tables = doc
            raw = _text_to_raw(md, d_tables, filename=path.name, doc_type=dtype,
                               model_label="docling", page_count=page_count)
            route = "Digital PDF → Docling"
            if not raw.line_items:
                warnings.append("Docling found no line items; retrying with the PDF text layer.")
                raw = None
        if raw is None:
            raw = _text_to_raw(text_layer, tl_tables, filename=path.name, doc_type=dtype,
                               model_label="pymupdf_text_layer", page_count=page_count)
            route = ("Digital PDF → PyMuPDF/pdfplumber text layer (Docling offline/unavailable)"
                     if doc is None else "Digital PDF → text-layer fallback")
        if raw.line_items:
            _attach_bboxes(raw, words)
        else:
            warnings.append("Text layer yielded no line items; falling back to the image route.")
            raw, dtype = None, DocumentType.SCANNED_PDF

    # ── Scanned PDF / image: OpenCV → VLM or Tesseract ──────────────────────
    if raw is None:
        raw_pages = rasterize_pdf(path, dpi=200) if path.suffix.lower() == ".pdf" else [_read_image_file(path)]
        pre = [preprocess_image(p) for p in raw_pages[:MAX_PAGES]]
        pages = [p.deskewed for p in pre]
        enhanced = [p.enhanced for p in pre]
        angle_note = ", ".join(f"p{i + 1}:{p.angle_deg:+.1f}°" for i, p in enumerate(pre))

        if use_vlm:
            try:
                data = _extract_json(_call_vlm(enhanced, _INVOICE_PROMPT))
                raw = _json_to_raw(data, filename=path.name, doc_type=dtype,
                                   model=f"{cfg.VLM_PROVIDER}:{getattr(cfg, cfg.VLM_PROVIDER.upper() + '_MODEL')}",
                                   pages=len(pages))
                route = f"{dtype.value} → OpenCV deskew+CLAHE ({angle_note}) → {cfg.VLM_PROVIDER.upper()} VLM"
            except Exception as exc:
                warnings.append(f"VLM extraction failed ({exc}); falling back to Tesseract.")
        if raw is None and tesseract_available():
            all_text, all_words = [], []
            for pno, img in enumerate(enhanced, start=1):
                t, w = _ocr_page(img, pno)
                all_text.append(t)
                all_words.extend(w)
            raw = _text_to_raw("\n".join(all_text), [], filename=path.name, doc_type=dtype,
                               model_label="tesseract", page_count=len(pages))
            _attach_bboxes(raw, all_words)
            route = f"{dtype.value} → OpenCV deskew+CLAHE ({angle_note}) → Tesseract OCR"
            if cfg.DEMO_MODE:
                warnings.append("DEMO_MODE=True: VLM skipped. Set DEMO_MODE=False and an API key for production accuracy.")
        if raw is None:
            raw = _failed(path.name, dtype, "No extractor available: set DEMO_MODE=False with a VLM API key, or install Tesseract.")
            warnings.append(raw.raw_text)
            route = "no extractor available"

    # ── DoubleTake ──────────────────────────────────────────────────────────
    if use_vlm and raw.vlm_model_used not in ("tesseract", "docling", "pymupdf_text_layer"):
        reader: RowReader = VLMReader()
    elif tesseract_available():
        sup, buy = raw.supplier_gstin, raw.buyer_gstin
        reader = OCRReader(bool(sup and buy and sup[:2] != buy[:2]))
    else:
        reader = RowReader()
    events = run_doubletake(raw, pages, reader, stem) if raw.line_items else []
    raw.processing_time_ms = round((time.perf_counter() - t0) * 1000, 1)
    return ExtractionOutput(raw, pages, events, str(path), route, warnings)


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from rich.console import Console

    con = Console()
    out = extract_invoice(demo_seed=42)
    con.print(f"[bold]Route:[/bold] {out.route}")
    con.print(f"Rows: {len(out.raw.line_items)} | status={out.raw.extraction_status} | "
              f"doubletake={out.raw.doubletake_triggered} (+{out.raw.doubletake_rows_recovered} rows)")
    for ev in out.events:
        con.print(f"  • {ev.kind} row {ev.row_no} [{ev.status}] {ev.reason}\n    crop: {ev.crop_path}")
    _write_png(TEMP_CROPS_DIR / "demo_page.png", out.page_images[0])
    con.print(f"Page image saved → {TEMP_CROPS_DIR / 'demo_page.png'}")
