"""
mock_data.py — Synthetic Indian MSME master data + realistic GST invoice generator.

Generates:
  • Party master  — 15 supplier companies with valid GSTINs across Indian states
  • SKU master    — 30 stock items (pharma, industrial, FMCG) with HSN codes
  • PO master     — 10 open purchase orders referencing the above
  • Mock invoices — Realistic RawExtractionResult objects (with optional error injection)

All data is deterministic (same seed = same output) so the UI demo is reproducible.
"""

from __future__ import annotations

import random
import uuid
from datetime import date, timedelta
from typing import Any

from config import cfg, INDIAN_STATE_CODES, validate_gstin_checksum, compute_gstin_checksum
from schemas import (
    PartyMaster,
    SKUMaster,
    PurchaseOrder,
    RawExtractionResult,
    RawLineItem,
    BoundingBox,
)

# ── Deterministic seed ────────────────────────────────────────────────────────
_RNG = random.Random(42)


# ═════════════════════════════════════════════════════════════════════════════
# Utility helpers
# ═════════════════════════════════════════════════════════════════════════════

def _build_gstin(state_code: str, pan: str) -> str:
    """
    Construct a syntactically valid GSTIN from a state code and PAN.
    Format: SS AAAAA 9999 A Z C
      SS    = 2-digit state code
      AAAAA = 5 PAN letters
      9999  = 4 PAN digits
      A     = last PAN letter
      1     = entity number for the PAN within the state
      Z     = default character
      C     = checksum character (computed)
    """
    pan = pan.upper().strip()
    # Build first 14 chars
    partial = f"{state_code}{pan}1Z"   # 13th char = entity number (1), 14th = 'Z'; checksum is the 15th
    checksum = compute_gstin_checksum(partial)
    return f"{partial}{checksum}"


def _random_pan(rng: random.Random) -> str:
    """Generate a syntactically valid but fictional PAN."""
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    entity_letters = "ABCFGHLJPTF"  # first char codes for different entity types
    # Format: AAAAA9999A
    return (
        rng.choice(entity_letters)
        + "".join(rng.choices(letters, k=4))
        + "".join(rng.choices("0123456789", k=4))
        + rng.choice(letters)
    )


def _random_date(rng: random.Random, days_back: int = 90) -> str:
    """Return ISO date string within last N days."""
    delta = rng.randint(0, days_back)
    return (date.today() - timedelta(days=delta)).isoformat()


def _fmt_inr(amount: float) -> str:
    """Format as Indian Rupees string with 2 decimal places."""
    return f"{amount:,.2f}"


# ═════════════════════════════════════════════════════════════════════════════
# Party Master
# ═════════════════════════════════════════════════════════════════════════════

_PARTY_SEEDS: list[dict[str, Any]] = [
    {"name": "Rajesh Chemicals Pvt Ltd",          "state_code": "27", "city": "Mumbai"},
    {"name": "BioPharm Industries Ltd",           "state_code": "29", "city": "Bengaluru"},
    {"name": "Surya Polymers & Resins Pvt Ltd",   "state_code": "24", "city": "Ahmedabad"},
    {"name": "Eastern Packing Solutions",          "state_code": "19", "city": "Kolkata"},
    {"name": "Delhi Precision Tools Mfg Co",      "state_code": "07", "city": "Delhi"},
    {"name": "Coimbatore Textiles Pvt Ltd",       "state_code": "33", "city": "Coimbatore"},
    {"name": "Hyderabad Pharma Supplies Ltd",     "state_code": "36", "city": "Hyderabad"},
    {"name": "Pune Auto Components Pvt Ltd",      "state_code": "27", "city": "Pune"},
    {"name": "Jaipur Handicrafts Exports Ltd",    "state_code": "08", "city": "Jaipur"},
    {"name": "Cochin Spice Traders Pvt Ltd",      "state_code": "32", "city": "Kochi"},
    {"name": "Ludhiana Steel Rolling Mills",       "state_code": "03", "city": "Ludhiana"},
    {"name": "Nashik Agro Products Pvt Ltd",      "state_code": "27", "city": "Nashik"},
    {"name": "Noida Electronics Mfg Ltd",         "state_code": "09", "city": "Noida"},
    {"name": "Surat Diamond & Jewellery Co",      "state_code": "24", "city": "Surat"},
    {"name": "Vizag Petrochemical Supplies",       "state_code": "28", "city": "Visakhapatnam"},
]


def build_party_master() -> list[PartyMaster]:
    """Return a list of 15 synthetic Indian supplier companies."""
    parties: list[PartyMaster] = []
    rng = random.Random(100)   # separate seed for stability

    for idx, seed in enumerate(_PARTY_SEEDS):
        pan = _random_pan(rng)
        gstin = _build_gstin(seed["state_code"], pan)
        party = PartyMaster(
            party_id=f"PARTY-{idx+1:03d}",
            name=seed["name"],
            gstin=gstin,
            state_code=seed["state_code"],
            address=f"{rng.randint(1, 999)}, Industrial Estate, {seed['city']}",
            pan=pan,
            email=f"accounts@{seed['name'].lower().replace(' ', '').replace('&', '')[:12]}.in",
            phone=f"+91-{rng.randint(70000, 99999)}{rng.randint(10000, 99999)}",
            is_active=True,
        )
        parties.append(party)

    return parties


# ═════════════════════════════════════════════════════════════════════════════
# SKU Master
# ═════════════════════════════════════════════════════════════════════════════

_SKU_SEEDS: list[dict[str, Any]] = [
    # Pharma
    {"code": "SKU-001", "name": "Paracetamol IP 500mg Tab",      "aliases": ["Paracetamol 500", "PCT Tab"],         "hsn": "30049099", "unit": "BOX",  "gst": 12.0, "rate": 285.00},
    {"code": "SKU-002", "name": "Metformin HCl 500mg",           "aliases": ["Metformin 500"],                      "hsn": "30049099", "unit": "BOX",  "gst": 12.0, "rate": 320.00},
    {"code": "SKU-003", "name": "Amoxicillin 250mg Capsules",    "aliases": ["Amox 250", "Amoxicillin Cap"],        "hsn": "29411010", "unit": "BOX",  "gst": 12.0, "rate": 460.00},
    {"code": "SKU-004", "name": "Surgical Spirit 450ml",         "aliases": ["Surg Spirit", "IPA 450ml"],           "hsn": "30059099", "unit": "BTL",  "gst": 12.0, "rate": 95.00},
    {"code": "SKU-005", "name": "Dextrose Injection 500ml",      "aliases": ["D5W 500ml", "Glucose Inj"],           "hsn": "30061090", "unit": "BAG",  "gst":  5.0, "rate": 48.00},
    # Industrial
    {"code": "SKU-006", "name": "HDPE Polymer Granules",         "aliases": ["HDPE Granules", "High Density PE"],  "hsn": "39011010", "unit": "KGS",  "gst": 18.0, "rate": 92.50},
    {"code": "SKU-007", "name": "Mild Steel HR Sheet 2mm",       "aliases": ["MS HR 2mm", "Steel Sheet"],           "hsn": "72082700", "unit": "KGS",  "gst": 18.0, "rate": 68.00},
    {"code": "SKU-008", "name": "Precision Bearing 6204ZZ",      "aliases": ["Ball Bearing 6204"],                  "hsn": "84821010", "unit": "NOS",  "gst": 18.0, "rate": 125.00},
    {"code": "SKU-009", "name": "Polyurethane Foam Sheet 25mm",  "aliases": ["PU Foam 25"],                         "hsn": "39211399", "unit": "SQM",  "gst": 18.0, "rate": 320.00},
    {"code": "SKU-010", "name": "Stainless Steel Fasteners M8",  "aliases": ["SS Fasteners M8", "M8 Bolt SS"],     "hsn": "73181599", "unit": "KGS",  "gst": 18.0, "rate": 285.00},
    # FMCG / Packing
    {"code": "SKU-011", "name": "BOPP Laminated Bags 10kg",      "aliases": ["BOPP Bags 10", "Lam Bags"],           "hsn": "39232990", "unit": "NOS",  "gst": 12.0, "rate": 8.50},
    {"code": "SKU-012", "name": "Corrugated Boxes 5-Ply A3",     "aliases": ["CB 5-Ply A3", "Corru Box A3"],       "hsn": "48191000", "unit": "NOS",  "gst": 12.0, "rate": 42.00},
    {"code": "SKU-013", "name": "Stretch Wrap Film 500mm",       "aliases": ["Stretch Film", "Pallet Wrap"],        "hsn": "39202090", "unit": "ROL",  "gst": 18.0, "rate": 1250.00},
    {"code": "SKU-014", "name": "Silica Gel Sachets 2g",         "aliases": ["Silica Sachet 2g"],                   "hsn": "28112290", "unit": "BAG",  "gst": 18.0, "rate": 950.00},
    {"code": "SKU-015", "name": "HDPE Drum 200L",                "aliases": ["Drum 200L", "Plastic Drum"],          "hsn": "39232090", "unit": "NOS",  "gst": 18.0, "rate": 1850.00},
    # Chemicals
    {"code": "SKU-016", "name": "Isopropyl Alcohol 99% AR",      "aliases": ["IPA 99%", "Isopropanol"],             "hsn": "29051220", "unit": "LTR",  "gst": 18.0, "rate": 85.00},
    {"code": "SKU-017", "name": "Sodium Hydroxide Flakes",       "aliases": ["NaOH Flakes", "Caustic Soda"],        "hsn": "28151100", "unit": "KGS",  "gst": 18.0, "rate": 42.00},
    {"code": "SKU-018", "name": "Hydrochloric Acid 35%",         "aliases": ["HCl 35%", "Muriatic Acid"],           "hsn": "28061010", "unit": "KGS",  "gst": 18.0, "rate": 28.00},
    {"code": "SKU-019", "name": "Acetone LR Grade 2.5L",         "aliases": ["Acetone 2.5L", "Propanone"],          "hsn": "29141100", "unit": "CAN",  "gst": 18.0, "rate": 480.00},
    {"code": "SKU-020", "name": "Methanol Technical Grade",      "aliases": ["Methanol TG", "Methyl Alcohol"],      "hsn": "29051100", "unit": "LTR",  "gst": 18.0, "rate": 38.00},
    # Engineering / MRO
    {"code": "SKU-021", "name": "V-Belt A-45 Industrial",        "aliases": ["V Belt A45"],                         "hsn": "40103100", "unit": "NOS",  "gst": 18.0, "rate": 185.00},
    {"code": "SKU-022", "name": "Oil Seal TC 35x52x8",           "aliases": ["Oil Seal 35-52"],                     "hsn": "84843000", "unit": "NOS",  "gst": 18.0, "rate": 95.00},
    {"code": "SKU-023", "name": "Compressed Air Filter 3/4\" BSP", "aliases": ["Air Filter 3/4"],                   "hsn": "84213990", "unit": "NOS",  "gst": 18.0, "rate": 1200.00},
    {"code": "SKU-024", "name": "Hydraulic Oil 46 ISO 20L",      "aliases": ["Hyd Oil 46", "ISO VG 46"],            "hsn": "27101990", "unit": "CAN",  "gst": 18.0, "rate": 2800.00},
    {"code": "SKU-025", "name": "Welding Electrode ER308L 3.2",  "aliases": ["Weld Elect 3.2", "SS Electrode"],    "hsn": "83110010", "unit": "PKT",  "gst": 18.0, "rate": 650.00},
    # Textiles
    {"code": "SKU-026", "name": "Cotton Grey Fabric 44\" Wide",  "aliases": ["Cotton Grey 44"],                     "hsn": "52081900", "unit": "MTR",  "gst":  5.0, "rate": 72.00},
    {"code": "SKU-027", "name": "Polyester Yarn 150D/48F",       "aliases": ["PET Yarn 150D"],                      "hsn": "54022000", "unit": "KGS",  "gst": 12.0, "rate": 160.00},
    # Agriculture
    {"code": "SKU-028", "name": "NPK Fertiliser 19:19:19",       "aliases": ["NPK 19-19-19"],                       "hsn": "31052000", "unit": "KGS",  "gst":  5.0, "rate": 28.00},
    {"code": "SKU-029", "name": "Neem Oil Cold Pressed 1L",      "aliases": ["Neem Oil 1L"],                        "hsn": "15155090", "unit": "BTL",  "gst":  5.0, "rate": 320.00},
    # Electronics
    {"code": "SKU-030", "name": "MCB 32A Single Pole",           "aliases": ["MCB 32A SP", "Circuit Breaker 32A"], "hsn": "85362000", "unit": "NOS",  "gst": 18.0, "rate": 285.00},
]


def build_sku_master() -> list[SKUMaster]:
    """Return a list of 30 synthetic SKUs."""
    return [
        SKUMaster(
            sku_code=s["code"],
            name=s["name"],
            aliases=s["aliases"],
            hsn_sac_code=s["hsn"],
            unit=s["unit"],
            gst_rate=s["gst"],
            standard_rate=s["rate"],
            is_active=True,
        )
        for s in _SKU_SEEDS
    ]


# ═════════════════════════════════════════════════════════════════════════════
# Purchase Order Master
# ═════════════════════════════════════════════════════════════════════════════

def build_po_master(
    parties: list[PartyMaster],
    skus: list[SKUMaster],
) -> list[PurchaseOrder]:
    """Generate 10 open POs referencing the party and SKU masters."""
    rng = random.Random(200)
    pos: list[PurchaseOrder] = []

    for i in range(10):
        party = rng.choice(parties)
        # 2–4 items per PO
        num_items = rng.randint(2, 4)
        selected_skus = rng.sample(skus, k=num_items)
        items = [
            {
                "sku_code": sku.sku_code,
                "qty": round(rng.uniform(5, 100), 2),
                "unit_price": round(sku.standard_rate * rng.uniform(0.95, 1.05), 2),
            }
            for sku in selected_skus
        ]
        pos.append(
            PurchaseOrder(
                po_number=f"PO-2026-{1001+i}",
                supplier_gstin=party.gstin,
                items=items,
                po_date=_random_date(rng, days_back=120),
                is_open=True,
            )
        )

    return pos


# ═════════════════════════════════════════════════════════════════════════════
# Mock Invoice Generator
# ═════════════════════════════════════════════════════════════════════════════

def _pick_line_items(
    rng: random.Random,
    skus: list[SKUMaster],
    num_items: int,
    supplier_state: str,
    buyer_state: str,
    inject_math_error: bool = False,
) -> tuple[list[RawLineItem], float, float, float, float]:
    """
    Generate realistic raw line items and compute totals.

    Returns:
        (line_items, subtotal, total_cgst, total_sgst, total_igst)
    """
    inter_state = supplier_state != buyer_state
    selected = rng.sample(skus, k=min(num_items, len(skus)))
    raw_items: list[RawLineItem] = []
    subtotal = 0.0
    total_cgst = 0.0
    total_sgst = 0.0
    total_igst = 0.0

    for idx, sku in enumerate(selected, start=1):
        qty = round(rng.uniform(2.0, 50.0), 2)
        unit_price = round(sku.standard_rate * rng.uniform(0.97, 1.03), 2)
        taxable = round(qty * unit_price, 2)

        gst_rate = sku.gst_rate
        half = round(gst_rate / 2, 1)

        if inter_state:
            igst_rate = gst_rate
            igst_amount = round(taxable * igst_rate / 100, 2)
            cgst_rate = cgst_amount = sgst_rate = sgst_amount = 0.0
        else:
            cgst_rate = sgst_rate = half
            cgst_amount = round(taxable * half / 100, 2)
            sgst_amount = cgst_amount  # always equal for intra-state
            igst_rate = igst_amount = 0.0

        line_total = round(taxable + cgst_amount + sgst_amount + igst_amount, 2)

        # Inject math error on first item (off by ₹1 on taxable amount)
        display_taxable = taxable
        if inject_math_error and idx == 1:
            display_taxable = round(taxable - 1.00, 2)

        subtotal += taxable
        total_cgst += cgst_amount
        total_sgst += sgst_amount
        total_igst += igst_amount

        raw_items.append(
            RawLineItem(
                sl_no=str(idx),
                description=sku.name,
                hsn_sac_code=sku.hsn_sac_code,
                quantity=str(qty),
                unit=sku.unit,
                unit_price=str(unit_price),
                discount_pct="0",
                discount_amount="0",
                taxable_amount=str(display_taxable),
                gst_rate=str(gst_rate),
                cgst_rate=str(cgst_rate),
                cgst_amount=str(cgst_amount),
                sgst_rate=str(sgst_rate),
                sgst_amount=str(sgst_amount),
                igst_rate=str(igst_rate),
                igst_amount=str(igst_amount),
                line_total=str(line_total),
            )
        )

    subtotal = round(subtotal, 2)
    total_cgst = round(total_cgst, 2)
    total_sgst = round(total_sgst, 2)
    total_igst = round(total_igst, 2)
    return raw_items, subtotal, total_cgst, total_sgst, total_igst


def _build_bboxes(items: list[RawLineItem], rng: random.Random) -> dict[str, dict]:
    """
    Generate plausible normalised bounding boxes for all extracted fields.
    Layout approximation based on a typical A4 invoice.
    """
    bboxes: dict[str, dict] = {}

    # Header fields — top 25% of page
    header_fields = {
        "supplier_name":    (0.05, 0.03, 0.45, 0.04),
        "supplier_gstin":   (0.05, 0.07, 0.35, 0.025),
        "supplier_address": (0.05, 0.095, 0.40, 0.04),
        "invoice_number":   (0.60, 0.03, 0.30, 0.025),
        "invoice_date":     (0.60, 0.06, 0.25, 0.025),
        "po_number":        (0.60, 0.09, 0.25, 0.025),
        "buyer_name":       (0.05, 0.16, 0.45, 0.04),
        "buyer_gstin":      (0.05, 0.20, 0.35, 0.025),
        "buyer_address":    (0.05, 0.225, 0.40, 0.04),
    }

    for field, (x, y, w, h) in header_fields.items():
        noise_x = rng.uniform(-0.005, 0.005)
        noise_y = rng.uniform(-0.003, 0.003)
        bboxes[field] = {
            "x": round(x + noise_x, 4),
            "y": round(y + noise_y, 4),
            "w": round(w, 4),
            "h": round(h, 4),
            "page": 1,
            "confidence": round(rng.uniform(0.88, 0.99), 3),
        }

    # Line item rows — distributed between y=0.30 and y=0.80
    table_start_y = 0.30
    row_height = 0.045
    for i, item in enumerate(items):
        y_pos = table_start_y + i * row_height
        bboxes[f"line_item_{i+1}"] = {
            "x": 0.03,
            "y": round(y_pos, 4),
            "w": 0.94,
            "h": row_height,
            "page": 1,
            "confidence": round(rng.uniform(0.82, 0.97), 3),
        }

    # Footer totals — bottom 20%
    footer_fields = {
        "subtotal":    (0.60, 0.82, 0.30, 0.025),
        "total_cgst":  (0.60, 0.845, 0.30, 0.025),
        "total_sgst":  (0.60, 0.870, 0.30, 0.025),
        "total_igst":  (0.60, 0.895, 0.30, 0.025),
        "grand_total": (0.55, 0.930, 0.35, 0.030),
    }

    for field, (x, y, w, h) in footer_fields.items():
        noise_y = rng.uniform(-0.002, 0.002)
        bboxes[field] = {
            "x": x,
            "y": round(y + noise_y, 4),
            "w": w,
            "h": h,
            "page": 1,
            "confidence": round(rng.uniform(0.90, 0.99), 3),
        }

    return bboxes


def generate_mock_invoice(
    parties: list[PartyMaster],
    skus: list[SKUMaster],
    pos: list[PurchaseOrder],
    seed: int | None = None,
    inject_errors: bool = True,
) -> RawExtractionResult:
    """
    Generate a single synthetic RawExtractionResult that mimics real VLM output.

    Args:
        parties:        Party master list
        skus:           SKU master list
        pos:            PO master list
        seed:           Optional RNG seed for reproducibility
        inject_errors:  Whether to inject math/OCR errors (from cfg settings)

    Returns:
        A RawExtractionResult ready to pass to the validator.
    """
    rng = random.Random(seed or _RNG.randint(0, 999999))

    supplier: PartyMaster = rng.choice(parties)
    buyer_state = cfg.COMPANY_STATE_CODE

    # Decide transaction type
    inter_state = supplier.state_code != buyer_state

    num_items = rng.randint(3, 7)

    # Error injection flags
    do_math_error = inject_errors and cfg.DEMO_INJECT_MATH_ERROR
    do_ocr_error  = inject_errors and cfg.DEMO_INJECT_OCR_ERROR
    do_missing_row = inject_errors and cfg.DEMO_INJECT_MISSING_ROW

    raw_items, subtotal, cgst, sgst, igst = _pick_line_items(
        rng, skus, num_items, supplier.state_code, buyer_state, do_math_error
    )

    # Simulate missing last row (DoubleTake trigger)
    cropped_items = raw_items
    hidden_last_item = None
    if do_missing_row and len(raw_items) >= 2:
        hidden_last_item = raw_items[-1]
        cropped_items = raw_items[:-1]
        # Totals intentionally still reflect ALL items (the invoice printed them)
        # so the validator/DoubleTake will detect the discrepancy

    total_tax = round(cgst + sgst + igst, 2)
    grand_total = round(subtotal + total_tax, 2)
    round_off = 0.00

    # OCR error injection: swap one digit in supplier GSTIN
    display_supplier_gstin = supplier.gstin
    if do_ocr_error:
        gstin_list = list(display_supplier_gstin)
        swap_pos = rng.randint(2, 13)
        original_char = gstin_list[swap_pos]
        # Swap a digit with another digit
        if original_char.isdigit():
            new_digit = str((int(original_char) + rng.randint(1, 8)) % 10)
            gstin_list[swap_pos] = new_digit
        display_supplier_gstin = "".join(gstin_list)

    # Find a matching PO (or none)
    matching_po = next(
        (p for p in pos if p.supplier_gstin == supplier.gstin and p.is_open),
        None,
    )
    po_number = matching_po.po_number if matching_po else f"PO-WALK-{rng.randint(100,999)}"

    inv_date = _random_date(rng, days_back=30)
    inv_number = f"INV/{supplier.state_code}/{rng.randint(10000, 99999)}/{date.today().year}"

    bboxes = _build_bboxes(cropped_items, rng)

    # Build raw text string (simulates VLM plain-text dump)
    raw_text_lines = [
        f"INVOICE",
        f"",
        f"Supplier: {supplier.name}",
        f"GSTIN: {display_supplier_gstin}",
        f"Address: {supplier.address}",
        f"",
        f"Invoice No: {inv_number}",
        f"Date: {inv_date}",
        f"PO Ref: {po_number}",
        f"",
        f"Bill To: {cfg.COMPANY_NAME}",
        f"GSTIN: {cfg.COMPANY_GSTIN}",
        f"",
        f"{'Sl':>3}  {'Description':<35} {'HSN':<10} {'Qty':>8} {'Rate':>10} {'Taxable':>12} {'GST%':>6} {'Tax':>10} {'Total':>12}",
        f"{'-'*110}",
    ]
    for item in cropped_items:
        raw_text_lines.append(
            f"{item.sl_no:>3}  {item.description:<35} {item.hsn_sac_code:<10} "
            f"{item.quantity:>8} {item.unit_price:>10} {item.taxable_amount:>12} "
            f"{item.gst_rate:>6}% {item.cgst_amount:>10} {item.line_total:>12}"
        )
    raw_text_lines += [
        f"",
        f"{'Subtotal':>80}: {_fmt_inr(subtotal):>15}",
        f"{'CGST':>80}: {_fmt_inr(cgst):>15}",
        f"{'SGST':>80}: {_fmt_inr(sgst):>15}",
        f"{'IGST':>80}: {_fmt_inr(igst):>15}",
        f"{'Grand Total':>80}: {_fmt_inr(grand_total):>15}",
    ]
    if hidden_last_item:
        raw_text_lines.append(f"\n[NOTE: Last row cropped in image — DoubleTake required]")

    return RawExtractionResult(
        source_filename=f"invoice_{inv_number.replace('/', '_')}.pdf",
        document_type="scanned_pdf" if rng.random() > 0.5 else "digital_pdf",
        extraction_status="partial" if do_missing_row else "success",
        vlm_model_used="demo_mock",
        page_count=rng.randint(1, 2),
        raw_text="\n".join(raw_text_lines),

        invoice_number=inv_number,
        invoice_date=inv_date,
        po_number=po_number,
        place_of_supply=INDIAN_STATE_CODES.get(buyer_state, buyer_state),

        supplier_name=supplier.name,
        supplier_gstin=display_supplier_gstin,
        supplier_address=supplier.address,
        supplier_state_code=supplier.state_code,
        supplier_pan=supplier.pan,

        buyer_name=cfg.COMPANY_NAME,
        buyer_gstin=cfg.COMPANY_GSTIN,
        buyer_address="123, Business Park, Andheri East, Mumbai 400069",
        buyer_state_code=buyer_state,

        line_items=cropped_items,

        subtotal=str(subtotal),
        total_cgst=str(cgst),
        total_sgst=str(sgst),
        total_igst=str(igst),
        total_tax=str(total_tax),
        round_off=str(round_off),
        grand_total=str(grand_total),
        grand_total_in_words=_amount_to_words(grand_total),

        bboxes=bboxes,
        doubletake_triggered=do_missing_row,
        doubletake_rows_recovered=1 if do_missing_row else 0,
        processing_time_ms=round(rng.uniform(800, 3500), 1),
    )


def generate_mock_invoice_batch(
    n: int = 5,
    inject_errors: bool = True,
) -> list[RawExtractionResult]:
    """
    Generate a batch of N synthetic invoices with varied error patterns.
    Useful for bulk-testing the pipeline.
    """
    parties = build_party_master()
    skus = build_sku_master()
    pos = build_po_master(parties, skus)

    return [
        generate_mock_invoice(parties, skus, pos, seed=i * 7 + 13, inject_errors=inject_errors)
        for i in range(n)
    ]


# ═════════════════════════════════════════════════════════════════════════════
# Amount to words (Indian numbering system)
# ═════════════════════════════════════════════════════════════════════════════

_ONES = [
    "", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine",
    "Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen",
    "Sixteen", "Seventeen", "Eighteen", "Nineteen",
]
_TENS = [
    "", "", "Twenty", "Thirty", "Forty", "Fifty",
    "Sixty", "Seventy", "Eighty", "Ninety",
]


def _words_lt_100(n: int) -> str:
    if n < 20:
        return _ONES[n]
    return (_TENS[n // 10] + (" " + _ONES[n % 10] if n % 10 else "")).strip()


def _words_lt_1000(n: int) -> str:
    if n < 100:
        return _words_lt_100(n)
    hundreds = n // 100
    rem = n % 100
    result = _ONES[hundreds] + " Hundred"
    if rem:
        result += " " + _words_lt_100(rem)
    return result


def _amount_to_words(amount: float) -> str:
    """
    Convert a float amount to Indian-style English words.
    E.g. 123456.78 → "Rupees One Lakh Twenty Three Thousand Four Hundred Fifty Six
                       and Seventy Eight Paise Only"
    """
    rupees = int(amount)
    paise = round((amount - rupees) * 100)

    def _convert(n: int) -> str:
        if n == 0:
            return "Zero"
        parts: list[str] = []
        crore = n // 10_000_000
        n %= 10_000_000
        lakh = n // 100_000
        n %= 100_000
        thousand = n // 1000
        n %= 1000
        if crore:
            parts.append(_words_lt_1000(crore) + " Crore")
        if lakh:
            parts.append(_words_lt_1000(lakh) + " Lakh")
        if thousand:
            parts.append(_words_lt_1000(thousand) + " Thousand")
        if n:
            parts.append(_words_lt_1000(n))
        return " ".join(parts)

    result = "Rupees " + _convert(rupees)
    if paise:
        result += f" and {_words_lt_100(paise)} Paise"
    return result + " Only"


# ═════════════════════════════════════════════════════════════════════════════
# Module-level pre-built masters (import directly for quick access)
# ═════════════════════════════════════════════════════════════════════════════

PARTY_MASTER: list[PartyMaster] = build_party_master()
SKU_MASTER: list[SKUMaster] = build_sku_master()
PO_MASTER: list[PurchaseOrder] = build_po_master(PARTY_MASTER, SKU_MASTER)


if __name__ == "__main__":
    # Quick smoke-test
    from rich.console import Console
    from rich.table import Table

    console = Console()

    console.print("\n[bold cyan]═══ Party Master (first 5) ═══[/bold cyan]")
    t = Table("ID", "Name", "GSTIN", "State", "Checksum OK?")
    for p in PARTY_MASTER[:5]:
        ok = validate_gstin_checksum(p.gstin)
        t.add_row(p.party_id, p.name[:30], p.gstin, p.state_code, "✅" if ok else "❌")
    console.print(t)

    console.print("\n[bold cyan]═══ SKU Master (first 5) ═══[/bold cyan]")
    t2 = Table("Code", "Name", "HSN", "GST%", "Std Rate")
    for s in SKU_MASTER[:5]:
        t2.add_row(s.sku_code, s.name[:40], s.hsn_sac_code, str(s.gst_rate), f"₹{s.standard_rate:,.2f}")
    console.print(t2)

    console.print("\n[bold cyan]═══ Sample Mock Invoice ═══[/bold cyan]")
    inv = generate_mock_invoice(PARTY_MASTER, SKU_MASTER, PO_MASTER, seed=42)
    console.print(f"  Invoice No : {inv.invoice_number}")
    console.print(f"  Supplier   : {inv.supplier_name}")
    console.print(f"  GSTIN      : {inv.supplier_gstin}")
    console.print(f"  Items      : {len(inv.line_items)}")
    console.print(f"  Grand Total: ₹{inv.grand_total}")
    console.print(f"  Status     : {inv.extraction_status}")
    console.print(f"  DoubleTake : {inv.doubletake_triggered}")
    console.print(f"\n[dim]{inv.raw_text[:400]}...[/dim]")
