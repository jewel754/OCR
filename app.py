"""
app.py — Streamlit HITL split-screen dashboard for the MSME GST Invoice Engine.

Run:  streamlit run app.py

Left  : invoice page with colour-coded bounding boxes (BoundingBox.to_pixel) + DoubleTake crops
Right : findings · inline edit + SKU-alias saving · audit-gated Tally XML download / HTTP push
"""

from __future__ import annotations

import copy
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from rapidfuzz import fuzz, process

from config import cfg, validate_gstin_checksum
from mock_data import PARTY_MASTER, PO_MASTER, SKU_MASTER
from router_extractor import ExtractionOutput, extract_invoice
from schemas import (
    Invoice,
    RawLineItem,
    ValidationError,
    ValidationResult,
    ValidationSeverity,
)
from tally_xml import (
    TallyXMLError,
    build_purchase_voucher_xml,
    push_to_tally,
    simulate_tally_push,
)
from validator import validate_invoice

st.set_page_config(page_title="MSME GST Invoice Engine", page_icon="🧾", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Findings that make a voucher structurally unsafe: fix the data (inline edit) — no override.
NON_OVERRIDABLE = {"STRUCT-001", "STRUCT-002", "STRUCT-003", "MATH-001", "MATH-007"}

COLORS = {
    "error":   (220, 38, 38),
    "warning": (245, 158, 11),
    "fixed":   (37, 99, 235),
    "ok":      (22, 163, 74),
}
RANK = {"ok": 0, "fixed": 1, "warning": 2, "error": 3}

ALIAS_FILE = cfg.MASTER_DIR / "sku_aliases.json"
AUDIT_FILE = cfg.OUTPUT_DIR / "audit_log.jsonl"
PUSHED_FILE = cfg.OUTPUT_DIR / "pushed_registry.json"

DEMO_SEEDS = [42, 13, 20, 27, 34, 41, 48, 55]

HEADER_FIELDS = [
    ("invoice_number", "Invoice number"), ("invoice_date", "Invoice date"), ("po_number", "PO number"),
    ("supplier_name", "Supplier name"), ("supplier_gstin", "Supplier GSTIN"), ("buyer_gstin", "Buyer GSTIN"),
]
TOTAL_FIELDS = ["subtotal", "total_cgst", "total_sgst", "total_igst", "total_tax", "round_off", "grand_total"]
LINE_COLS = [
    "sl_no", "description", "hsn_sac_code", "quantity", "unit", "unit_price", "discount_pct", "discount_amount",
    "taxable_amount", "gst_rate", "cgst_rate", "cgst_amount", "sgst_rate", "sgst_amount",
    "igst_rate", "igst_amount", "line_total",
]


# ─────────────────────────────────────────────────────────────────────────────
# Small UI helpers
# ─────────────────────────────────────────────────────────────────────────────

def show_image(img: Any, caption: str | None = None) -> None:
    """st.image across Streamlit versions (use_container_width vs use_column_width)."""
    try:
        st.image(img, caption=caption, use_container_width=True)
    except TypeError:
        st.image(img, caption=caption, use_column_width=True)


def bgr_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "LiberationSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


# ─────────────────────────────────────────────────────────────────────────────
# Masters, aliases, audit, registry
# ─────────────────────────────────────────────────────────────────────────────

def _load_aliases() -> dict[str, list[str]]:
    if ALIAS_FILE.exists():
        try:
            return json.loads(ALIAS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_alias(sku_code: str, alias: str) -> bool:
    """Persist a description → SKU alias and apply it to the live master. Returns True if new."""
    alias = alias.strip()
    if not alias:
        return False
    for sku in st.session_state.skus:
        if sku.sku_code == sku_code:
            if alias.lower() in (n.lower() for n in sku.all_names):
                return False
            sku.aliases.append(alias)
            break
    data = _load_aliases()
    data.setdefault(sku_code, [])
    if alias not in data[sku_code]:
        data[sku_code].append(alias)
    ALIAS_FILE.parent.mkdir(parents=True, exist_ok=True)
    ALIAS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return True


def _init_state() -> None:
    if "skus" not in st.session_state:
        skus = copy.deepcopy(SKU_MASTER)
        for sku in skus:
            for alias in _load_aliases().get(sku.sku_code, []):
                if alias.lower() not in (n.lower() for n in sku.all_names):
                    sku.aliases.append(alias)
        st.session_state.skus = skus
    st.session_state.setdefault("pos", PO_MASTER)
    st.session_state.setdefault("ctx", None)
    st.session_state.setdefault("push_result", None)


def log_audit(action: str, inv: Invoice, extra: dict[str, Any]) -> None:
    AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "action": action,
        "invoice_number": inv.invoice_number,
        "supplier_gstin": inv.supplier.gstin,
        "supplier_name": inv.supplier.name,
        "grand_total": inv.totals.grand_total,
        **extra,
    }
    with AUDIT_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _registry() -> dict[str, Any]:
    if PUSHED_FILE.exists():
        try:
            return json.loads(PUSHED_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _dup_key(inv: Invoice) -> str:
    return f"{inv.supplier.gstin}|{inv.invoice_number}".upper()


def _register_push(inv: Invoice, voucher: str) -> None:
    reg = _registry()
    reg[_dup_key(inv)] = {"ts": datetime.now().isoformat(timespec="seconds"), "voucher": voucher}
    PUSHED_FILE.parent.mkdir(parents=True, exist_ok=True)
    PUSHED_FILE.write_text(json.dumps(reg, indent=2), encoding="utf-8")


def resolve_party_ledger(inv: Invoice) -> str:
    """Map the extracted supplier to the exact Tally ledger name (GSTIN first, then fuzzy name)."""
    for p in PARTY_MASTER:
        if p.gstin == inv.supplier.gstin:
            return p.name
    best = process.extractOne(inv.supplier.name, [p.name for p in PARTY_MASTER], scorer=fuzz.token_sort_ratio)
    if best and best[1] >= cfg.PARTY_MATCH_MIN_SCORE:
        return best[0]
    return inv.supplier.name


def suggest_gstin(bad: str) -> tuple[str, str] | None:
    """Closest checksum-valid master GSTIN within Hamming distance ≤ 2 (typical OCR digit swaps)."""
    best: tuple[int, str, str] | None = None
    for p in PARTY_MASTER:
        if len(p.gstin) != len(bad) or not validate_gstin_checksum(p.gstin):
            continue
        dist = sum(a != b for a, b in zip(p.gstin, bad))
        if dist <= 2 and (best is None or dist < best[0]):
            best = (dist, p.gstin, p.name)
    return (best[1], best[2]) if best else None


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _validate(raw) -> ValidationResult:
    return validate_invoice(raw, st.session_state.skus, st.session_state.pos)


def run_pipeline(out: ExtractionOutput) -> None:
    raw = out.raw.model_copy(deep=True)
    st.session_state.ctx = {"out": out, "raw": raw, "result": _validate(raw), "ver": 0}
    st.session_state.push_result = None


def revalidate() -> None:
    ctx = st.session_state.ctx
    ctx["result"] = _validate(ctx["raw"])
    ctx["ver"] += 1
    st.session_state.push_result = None


# ─────────────────────────────────────────────────────────────────────────────
# Bounding-box overlay
# ─────────────────────────────────────────────────────────────────────────────

_HEADER_MAP = {
    "supplier.gstin": "supplier_gstin", "supplier.state_code": "supplier_gstin", "supplier.name": "supplier_name",
    "buyer.gstin": "buyer_gstin", "invoice_number": "invoice_number", "invoice_date": "invoice_date",
    "po_number": "po_number",
}


def _error_key(field: str, inv: Invoice) -> str | None:
    m = re.match(r"line_items\[(\d+)\]", field)
    if m:
        sl = int(m.group(1))
        for pos, li in enumerate(inv.line_items, start=1):
            if li.sl_no == sl:
                return f"line_item_{pos}"
        return None
    if field.startswith("totals."):
        return field.split(".", 1)[1]
    return _HEADER_MAP.get(field)


def field_status(inv: Invoice, result: ValidationResult) -> tuple[dict[str, str], dict[str, list[str]]]:
    status: dict[str, str] = {}
    codes: dict[str, list[str]] = {}
    for e in result.errors:
        key = _error_key(e.field, inv)
        if not key:
            continue
        if e.auto_corrected:
            st_ = "fixed"
        elif e.severity == ValidationSeverity.ERROR:
            st_ = "error"
        elif e.severity == ValidationSeverity.WARNING:
            st_ = "warning"
        else:
            continue
        if RANK[st_] > RANK.get(status.get(key, "ok"), 0):
            status[key] = st_
        codes.setdefault(key, []).append(e.code)
    return status, codes


def render_overlay(img_bgr: np.ndarray, inv: Invoice, result: ValidationResult, page: int, show: bool) -> Image.Image:
    base = bgr_to_pil(img_bgr).convert("RGBA")
    if not show:
        return base.convert("RGB")
    W, H = base.size
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    font = _font(max(13, W // 85))
    status, codes = field_status(inv, result)

    boxes: list[tuple[str, Any]] = list(inv.bboxes.fields.items())
    for pos, li in enumerate(inv.line_items, start=1):
        if li.bbox is not None:
            boxes.append((f"line_item_{pos}", li.bbox))

    for key, bb in boxes:
        if bb.page != page:
            continue
        x, y, w, h = bb.to_pixel(W, H)
        s = status.get(key, "ok")
        r, g, b = COLORS[s]
        d.rectangle([x, y, x + w, y + h], fill=(r, g, b, 38), outline=(r, g, b, 235), width=3)
        if s != "ok":
            tag = ",".join(dict.fromkeys(codes.get(key, [])))
            tw = d.textlength(tag, font=font)
            ty = max(0, y - font.size - 4) if hasattr(font, "size") else max(0, y - 16)
            d.rectangle([x, ty, x + tw + 8, ty + (font.size if hasattr(font, "size") else 14) + 4], fill=(r, g, b, 235))
            d.text((x + 4, ty + 1), tag, fill=(255, 255, 255, 255), font=font)
    return Image.alpha_composite(base, layer).convert("RGB")


def legend() -> None:
    items = [("ok", "Clean"), ("fixed", "Auto-corrected"), ("warning", "Warning"), ("error", "Blocking error")]
    html = "".join(
        f"<span style='display:inline-block;margin-right:14px'><span style='display:inline-block;width:12px;"
        f"height:12px;border-radius:2px;background:rgb{COLORS[k]};margin-right:5px'></span>{label}</span>"
        for k, label in items
    )
    st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def sidebar() -> None:
    sb = st.sidebar
    sb.title("🧾 MSME GST Engine")
    sb.caption(f"Company: **{cfg.COMPANY_NAME}**  \nGSTIN `{cfg.COMPANY_GSTIN}`")
    mode = "🟢 DEMO_MODE (offline, synthetic)" if cfg.DEMO_MODE else f"🔵 LIVE · {cfg.VLM_PROVIDER.upper()}"
    sb.markdown(f"**Mode:** {mode}")
    if not cfg.DEMO_MODE and not cfg.vlm_is_configured():
        sb.warning("No VLM API key set — scanned documents fall back to Tesseract.")
    sb.divider()

    options = ["Demo invoice", "Upload PDF / image"] if cfg.DEMO_MODE else ["Upload PDF / image", "Demo invoice"]
    source = sb.radio("Source", options, key="source_kind")

    if source == "Demo invoice":
        if not cfg.DEMO_MODE:
            sb.info("Demo invoices need DEMO_MODE=True.")
        seed = sb.selectbox("Demo seed (different suppliers / error mixes)", DEMO_SEEDS, key="demo_seed")
        if sb.button("▶ Process demo invoice", type="primary", disabled=not cfg.DEMO_MODE, key="btn_demo"):
            with st.spinner("Extracting → DoubleTake → validating…"):
                run_pipeline(extract_invoice(demo_seed=int(seed)))
            st.rerun()
    else:
        up = sb.file_uploader("Invoice (PDF / JPG / PNG)", type=["pdf", "jpg", "jpeg", "png", "tif", "tiff"], key="uploader")
        if sb.button("▶ Process upload", type="primary", disabled=up is None, key="btn_upload"):
            dest = cfg.UPLOAD_DIR / Path(up.name).name
            dest.write_bytes(up.getvalue())
            with st.spinner("Routing → extracting → DoubleTake → validating…"):
                out = extract_invoice(dest)
                run_pipeline(out)
            st.rerun()

    sb.divider()
    sb.caption(f"Tally endpoint: `{cfg.tally_url()}` · company `{cfg.TALLY_COMPANY_NAME}`")
    sb.caption(f"Audit log: `{AUDIT_FILE.name}` in outputs/")


# ─────────────────────────────────────────────────────────────────────────────
# Left pane: invoice image + DoubleTake
# ─────────────────────────────────────────────────────────────────────────────

def left_pane(ctx: dict[str, Any]) -> None:
    out: ExtractionOutput = ctx["out"]
    result: ValidationResult = ctx["result"]
    inv = result.corrected_invoice
    st.subheader("Invoice viewer")
    if not out.page_images or inv is None:
        st.error("No page image available for this document.")
        return

    c1, c2 = st.columns([1, 1])
    show = c1.toggle("Show bounding boxes", value=True, key="show_boxes")
    page = 1
    if len(out.page_images) > 1:
        page = c2.number_input("Page", 1, len(out.page_images), 1, key="page_no")
    legend()
    show_image(render_overlay(out.page_images[int(page) - 1], inv, result, int(page), show))

    st.subheader("DoubleTake re-verification")
    if not out.events:
        st.success("No cropped rows or row-math failures detected — DoubleTake not triggered.")
        return
    st.caption(f"{len(out.events)} event(s) · crops saved under `temp_crops/`")
    for i, ev in enumerate(out.events, start=1):
        icon = {"recovered": "✅", "unresolved": "⚠️", "flagged": "🚩"}.get(ev.status, "•")
        kind = "Missing / cropped row" if ev.kind == "missing_row" else "Row math mismatch"
        with st.expander(f"{icon} {i}. {kind} — row {ev.row_no} · {ev.status.upper()}", expanded=(i == 1)):
            st.write(ev.reason)
            a, b, c = st.columns([1, 1, 1])
            if ev.orig_path and Path(ev.orig_path).exists():
                with a:
                    st.caption("Original page region")
                    show_image(ev.orig_path)
            if ev.crop_path and Path(ev.crop_path).exists():
                with b:
                    st.caption("Zoomed + CLAHE re-read crop")
                    show_image(ev.crop_path)
            with c:
                st.caption(f"Re-read by `{ev.reader or 'n/a'}`")
                diff = pd.DataFrame(
                    {"before": pd.Series(ev.before), "after": pd.Series(ev.after)}
                ).fillna("")
                st.dataframe(diff, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Right pane tabs
# ─────────────────────────────────────────────────────────────────────────────

def findings_tab(ctx: dict[str, Any]) -> None:
    result: ValidationResult = ctx["result"]
    inv = result.corrected_invoice
    assert inv is not None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Blocking errors", len(result.blocking_errors))
    c2.metric("Warnings", result.warning_count)
    c3.metric("Auto-corrected", result.auto_corrected_count)
    c4.metric("Line items", inv.line_item_count)

    if result.blocking_errors:
        st.error("Tally push is blocked until these are fixed or explicitly overridden (see **Tally** tab).")
    else:
        st.success("No blocking errors — voucher can be exported.")

    rows = [
        {
            "Code": e.code,
            "Severity": e.severity.value.upper(),
            "Blocking": "⛔" if e.blocks_push else "",
            "Field": e.field,
            "Expected": e.expected,
            "Found": e.found,
            "Auto-fixed": "✅" if e.auto_corrected else "",
            "Message": e.message,
        }
        for e in result.errors
    ]
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    gst002 = [e for e in result.errors if e.code == "GST-002"]
    if gst002:
        sug = suggest_gstin(inv.supplier.gstin)
        if sug:
            st.info(f"Closest checksum-valid master GSTIN: `{sug[0]}` — **{sug[1]}**")
            if st.button("Apply suggested GSTIN & re-validate", key=f"fixgstin_{ctx['ver']}"):
                ctx["raw"].supplier_gstin = sug[0]
                revalidate()
                st.rerun()
        else:
            st.warning("No master GSTIN within 2 characters. Verify on the GST portal, then correct inline.")


def edit_tab(ctx: dict[str, Any]) -> None:
    raw = ctx["raw"]
    ver = ctx["ver"]
    st.caption("Edit what the extractor read, then re-validate. Auto-corrections are recomputed from your values.")

    with st.form(f"hdr_form_{ver}"):
        cols = st.columns(2)
        new_vals: dict[str, str] = {}
        for i, (name, label) in enumerate(HEADER_FIELDS):
            new_vals[name] = cols[i % 2].text_input(label, value=getattr(raw, name), key=f"h_{name}_{ver}")
        st.markdown("**Totals**")
        tcols = st.columns(4)
        for i, name in enumerate(TOTAL_FIELDS):
            new_vals[name] = tcols[i % 4].text_input(name, value=getattr(raw, name), key=f"t_{name}_{ver}")
        submitted_hdr = st.form_submit_button("Apply header & totals · re-validate")
    if submitted_hdr:
        for k, v in new_vals.items():
            setattr(raw, k, v.strip())
        raw.supplier_state_code = raw.supplier_gstin[:2] if len(raw.supplier_gstin) >= 2 else raw.supplier_state_code
        revalidate()
        st.rerun()

    st.markdown("**Line items** (add / delete rows with the table controls)")
    df = pd.DataFrame([{c: getattr(li, c, "") for c in LINE_COLS} for li in raw.line_items], columns=LINE_COLS)
    edited = st.data_editor(df, num_rows="dynamic", use_container_width=True, key=f"line_editor_{ver}")
    if st.button("Apply line edits · re-validate", key=f"apply_lines_{ver}"):
        items: list[RawLineItem] = []
        for _, row in edited.iterrows():
            rec = {c: ("" if pd.isna(row[c]) else str(row[c]).strip()) for c in LINE_COLS}
            if not rec["description"]:
                continue
            rec["sl_no"] = rec["sl_no"] or str(len(items) + 1)
            rec["discount_pct"] = rec["discount_pct"] or "0"
            rec["discount_amount"] = rec["discount_amount"] or "0"
            items.append(RawLineItem(**rec))
        raw.line_items = items
        revalidate()
        st.rerun()

    st.divider()
    st.markdown("**Reconciliation & SKU alias saving**")
    inv = ctx["result"].corrected_invoice
    assert inv is not None
    recon = pd.DataFrame(
        [
            {
                "Sl": li.sl_no, "Description": li.description, "Matched SKU": li.matched_sku_code or "—",
                "Match %": round(li.sku_match_score), "PO Δ%": li.po_rate_deviation_pct,
                "Taxable ₹": li.taxable_amount, "Tax ₹": li.tax.total_tax, "Line ₹": li.line_total,
            }
            for li in inv.line_items
        ]
    )
    st.dataframe(recon, use_container_width=True, hide_index=True)

    skus = st.session_state.skus
    sku_labels = [f"{s.sku_code} · {s.name}" for s in skus]
    unmatched = [li for li in inv.line_items if not li.matched_sku_code]
    candidates = unmatched or inv.line_items
    if unmatched:
        st.warning(f"{len(unmatched)} line(s) could not be matched to the SKU master — map them below.")
    if candidates:
        a, b = st.columns([2, 2])
        line_choice = a.selectbox(
            "Line", [f"{li.sl_no} · {li.description}" for li in candidates], key=f"alias_line_{ver}"
        )
        sku_choice = b.selectbox("Map to SKU", sku_labels, key=f"alias_sku_{ver}")
        if st.button("💾 Save SKU alias & re-validate", key=f"alias_save_{ver}"):
            description = line_choice.split(" · ", 1)[1]
            code = sku_choice.split(" · ", 1)[0]
            if _save_alias(code, description):
                st.toast(f"Alias saved: “{description}” → {code}")
            else:
                st.toast("Alias already exists for that SKU.")
            revalidate()
            st.rerun()


def tally_tab(ctx: dict[str, Any]) -> None:
    result: ValidationResult = ctx["result"]
    inv = result.corrected_invoice
    assert inv is not None
    ver = ctx["ver"]
    blocking = result.blocking_errors
    hard = [e for e in blocking if e.code in NON_OVERRIDABLE]
    soft = [e for e in blocking if e.code not in NON_OVERRIDABLE]

    # ── Override gate ───────────────────────────────────────────────────────
    override_ok = False
    reason, approver = "", ""
    if blocking:
        st.error("⛔ **Blocking errors present**")
        for e in blocking:
            tag = "fix data" if e.code in NON_OVERRIDABLE else "overridable"
            st.markdown(f"- `{e.code}` ({tag}) — {e.message}")
        if hard:
            st.warning("Some blockers cannot be overridden (voucher would be structurally unsafe). "
                       "Correct them in the **Edit** tab and re-validate.")
        elif soft:
            st.markdown("**Force-approve (audit-logged)**")
            ov = st.checkbox("Override Blocking Error (Log Audit Trail)", key=f"override_{ver}")
            if ov:
                c1, c2 = st.columns(2)
                approver = c1.text_input("Approver name", key=f"approver_{ver}").strip()
                reason = c2.text_input("Audit reason (min 10 characters)", key=f"reason_{ver}").strip()
                if len(reason) >= 10 and approver:
                    override_ok = True
                else:
                    st.info("Enter the approver and an audit reason (≥ 10 chars) to unlock export.")
    can_export = (not blocking) or override_ok

    # ── Voucher XML ─────────────────────────────────────────────────────────
    st.markdown("**Voucher**")
    party_default = resolve_party_ledger(inv)
    party_name = st.text_input("Party ledger name in Tally", value=party_default, key=f"party_{ver}")
    narration = ""
    if override_ok:
        codes = ",".join(sorted({e.code for e in blocking}))
        narration = f"OVERRIDE[{codes}] approved by {approver}: {reason}"

    xml, xml_error = "", ""
    if can_export:
        try:
            xml = build_purchase_voucher_xml(inv, party_ledger_name=party_name, narration=narration)
        except (TallyXMLError, AssertionError) as exc:
            xml_error = str(exc)
    if xml_error:
        st.error(f"Cannot build a balanced voucher: {xml_error}")

    audit_extra = {
        "overridden_codes": sorted({e.code for e in blocking}) if override_ok else [],
        "override_reason": reason if override_ok else "",
        "approver": approver if override_ok else "",
    }

    if xml:
        st.success("Voucher balances to 0.00 (party credit = Σ debits).")
        with st.expander("XML preview", expanded=False):
            st.code(xml, language="xml")

        fname = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"tally_purchase_{inv.invoice_number}.xml")
        st.download_button(
            "⬇ Download Tally XML", data=xml.encode("utf-8"), file_name=fname, mime="application/xml",
            key=f"dl_{ver}", on_click=log_audit, args=("xml_download", inv, audit_extra),
        )

        st.markdown(f"**Push to TallyPrime** · `{cfg.tally_url()}`")
        live = st.checkbox(
            "Send to live Tally (otherwise simulated, nothing leaves this machine)",
            value=not cfg.DEMO_MODE, key=f"live_{ver}",
        )
        dup = _registry().get(_dup_key(inv))
        dup_ack = True
        if dup:
            st.warning(f"This supplier invoice was already pushed on {dup['ts']} (voucher {dup['voucher']}).")
            dup_ack = st.checkbox("I confirm this is NOT a duplicate", key=f"dup_{ver}")

        if st.button("🚀 Push to Tally (localhost:9000)", type="primary", disabled=not dup_ack, key=f"push_{ver}"):
            with st.spinner("Posting voucher…"):
                res = (push_to_tally(xml, invoice_id=inv.invoice_id) if live
                       else simulate_tally_push(xml, invoice_id=inv.invoice_id))
            st.session_state.push_result = (res, live)
            log_audit("tally_push" if live else "tally_push_simulated", inv,
                      {**audit_extra, "success": res.success, "voucher": res.tally_voucher_number,
                       "error": res.error_message})
            if res.success and live:
                _register_push(inv, res.tally_voucher_number)

        pr = st.session_state.push_result
        if pr:
            res, was_live = pr
            if res.success:
                st.success(f"{'Posted to Tally' if was_live else 'Simulated push OK'} — voucher ref "
                           f"`{res.tally_voucher_number}` at {res.pushed_at}")
            else:
                st.error(res.error_message or "Push failed.")
            with st.expander("Tally response"):
                st.code(res.tally_response_xml or "(empty)", language="xml")
    elif not blocking or override_ok:
        pass
    else:
        st.info("Export and push are locked until blocking errors are resolved.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    _init_state()
    sidebar()
    st.title("Deterministic GST Invoice Ingestion → TallyPrime")

    ctx = st.session_state.ctx
    if ctx is None:
        st.info("Pick a source in the sidebar and press **Process**. "
                "In DEMO_MODE the pipeline injects a GSTIN OCR error, a ₹1 row slip and a cropped last row "
                "so you can watch DoubleTake, the validator and the override gate work end to end.")
        return

    out: ExtractionOutput = ctx["out"]
    raw = ctx["raw"]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Extraction", raw.extraction_status)
    m2.metric("Model", raw.vlm_model_used)
    m3.metric("DoubleTake rows recovered", raw.doubletake_rows_recovered)
    m4.metric("Time", f"{raw.processing_time_ms / 1000:.1f}s")
    m5.metric("Pages", len(out.page_images))
    st.caption(f"Route: {out.route}")
    for w in out.warnings:
        st.warning(w)

    if ctx["result"].corrected_invoice is None:
        st.error("Validation produced no invoice.")
        return

    left, right = st.columns([5, 6], gap="large")
    with left:
        left_pane(ctx)
    with right:
        t1, t2, t3 = st.tabs(["🔎 Findings", "✏️ Edit & SKU aliases", "📤 Tally"])
        with t1:
            findings_tab(ctx)
        with t2:
            edit_tab(ctx)
        with t3:
            tally_tab(ctx)


main()
