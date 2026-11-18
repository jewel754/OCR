"""
tally_xml.py — Validated Invoice  →  TallyPrime HTTP-XML Purchase Voucher.

Strict double-entry sign convention (Tally "Import Data" semantics)
───────────────────────────────────────────────────────────────────
  Party ledger        <LEDGERENTRIES.LIST>          ISDEEMEDPOSITIVE=No   AMOUNT = +grand_total   (Credit)
  Purchase / stock    <ALLINVENTORYENTRIES.LIST>    ISDEEMEDPOSITIVE=Yes  AMOUNT = -taxable       (Debit)
  Input CGST/SGST/IGST<LEDGERENTRIES.LIST>          ISDEEMEDPOSITIVE=Yes  AMOUNT = -tax_amount    (Debit)
  Round-off (if any)  <LEDGERENTRIES.LIST>          debit  → Yes / -amt ;  credit → No / +amt

  party_amt + Σ(debits) == 0.00   is asserted BEFORE the XML string is returned,
  and the finished XML is re-parsed and re-summed independently (verify_voucher_balanced).

Public API
──────────
  build_purchase_voucher_xml(invoice, party_ledger_name=None, narration="") -> str
  verify_voucher_balanced(xml) -> Decimal
  push_to_tally(xml, ...)            -> TallyPushResult     (HTTP POST to localhost:9000)
  simulate_tally_push(xml, ...)      -> TallyPushResult     (offline DEMO response)
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP

import httpx
from lxml import etree

from config import cfg, INDIAN_STATE_CODES
from schemas import Invoice, TallyPushResult

_TWO = Decimal("0.01")
_ZERO = Decimal("0.00")
_DATE_FORMATS = ["%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y", "%d %b %Y", "%d %B %Y"]
_XML_ILLEGAL = re.compile("[^\u0009\u000A\u000D\u0020-\uD7FF\uE000-\uFFFD]")


class TallyXMLError(ValueError):
    """Raised when a voucher cannot be built safely (never push a doubtful voucher)."""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _d(value: float | int | str | Decimal) -> Decimal:
    """Exact 2-dp Decimal (half-up) from any numeric input."""
    return Decimal(str(value)).quantize(_TWO, rounding=ROUND_HALF_UP)


def _amt(value: Decimal) -> str:
    """Decimal → Tally amount string, e.g. '-2850.00'. Never emits '-0.00'."""
    value = value.quantize(_TWO, rounding=ROUND_HALF_UP)
    if value == 0:
        value = _ZERO
    return format(value, "f")


def _clean(text: str) -> str:
    return _XML_ILLEGAL.sub("", str(text or "")).strip()


def _tally_date(date_str: str) -> str:
    """Any supported invoice date format → YYYYMMDD."""
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(date_str.strip(), fmt).strftime("%Y%m%d")
        except (ValueError, AttributeError):
            continue
    raise TallyXMLError(f"Invoice date {date_str!r} cannot be converted to YYYYMMDD.")


def _qty_str(qty: float, unit: str) -> str:
    return f" {qty:.2f} {unit}".rstrip() if unit else f" {qty:.2f}"


def _sub(parent: etree._Element, tag: str, text: str | None = None, **attrs: str) -> etree._Element:
    el = etree.SubElement(parent, tag, **attrs)
    if text is not None:
        el.text = _clean(text)
    return el


def _state_name(code: str, fallback: str = "") -> str:
    return INDIAN_STATE_CODES.get(code, fallback or code)


# ─────────────────────────────────────────────────────────────────────────────
# Voucher builder
# ─────────────────────────────────────────────────────────────────────────────

def build_purchase_voucher_xml(
    invoice: Invoice,
    *,
    party_ledger_name: str | None = None,
    narration: str = "",
    company_name: str | None = None,
) -> str:
    """
    Convert a validated Invoice into a TallyPrime <ENVELOPE> purchase voucher.

    Raises TallyXMLError when the voucher cannot be made to balance within
    GRAND_TOTAL_TOLERANCE, or mandatory data (date, party, lines) is missing.
    """
    if not invoice.line_items:
        raise TallyXMLError("Invoice has no line items.")

    party_name = _clean(party_ledger_name or invoice.supplier.name)
    if not party_name:
        raise TallyXMLError("Party ledger name is empty.")
    if not invoice.invoice_number.strip():
        raise TallyXMLError("Invoice number is empty.")

    vdate = _tally_date(invoice.invoice_date)
    company = _clean(company_name or cfg.TALLY_COMPANY_NAME)

    # ── 1. Amounts (exact Decimal arithmetic) ───────────────────────────────
    line_taxables = [_d(li.taxable_amount) for li in invoice.line_items]
    sum_taxable = sum(line_taxables, _ZERO)
    cgst = sum((_d(li.tax.cgst_amount) for li in invoice.line_items), _ZERO)
    sgst = sum((_d(li.tax.sgst_amount) for li in invoice.line_items), _ZERO)
    igst = sum((_d(li.tax.igst_amount) for li in invoice.line_items), _ZERO)
    grand = _d(invoice.totals.grand_total)

    if grand <= 0:
        raise TallyXMLError("Grand total must be greater than zero.")

    residual = grand - (sum_taxable + cgst + sgst + igst)   # = round-off ledger (signed)
    if abs(residual) > _d(cfg.GRAND_TOTAL_TOLERANCE):
        raise TallyXMLError(
            f"Grand total ₹{grand} does not tie to lines (taxable ₹{sum_taxable} + tax "
            f"₹{cgst + sgst + igst}); residual ₹{residual} exceeds tolerance ₹{cfg.GRAND_TOTAL_TOLERANCE}. "
            "Correct the invoice before generating the voucher."
        )

    party_amt: Decimal = grand                                   # Credit  (+)
    sum_debits: Decimal = _ZERO                                  # Debits  (−)

    tax_entries: list[tuple[str, Decimal]] = []
    for ledger, value in (
        (cfg.TALLY_LEDGER_CGST, cgst),
        (cfg.TALLY_LEDGER_SGST, sgst),
        (cfg.TALLY_LEDGER_IGST, igst),
    ):
        if value != 0:
            tax_entries.append((ledger, -value))                 # Debit → negative

    for taxable in line_taxables:
        sum_debits += -taxable
    for _, signed in tax_entries:
        sum_debits += signed

    roundoff_ledger = getattr(cfg, "TALLY_LEDGER_ROUNDOFF", "Round Off")
    roundoff_signed = _ZERO
    if residual != 0:
        roundoff_signed = -residual          # residual>0 → debit (−) ; residual<0 → credit (+)
        sum_debits += roundoff_signed

    # Built-in double-entry assertion (raises even under `python -O`)
    if not (round(party_amt + sum_debits, 2) == 0.0):
        raise AssertionError(
            f"Voucher out of balance: party {party_amt} + debits {sum_debits} "
            f"= {party_amt + sum_debits}"
        )

    # ── 2. XML tree ─────────────────────────────────────────────────────────
    env = etree.Element("ENVELOPE")
    header = _sub(env, "HEADER")
    _sub(header, "TALLYREQUEST", "Import Data")
    body = _sub(env, "BODY")
    importdata = _sub(body, "IMPORTDATA")
    reqdesc = _sub(importdata, "REQUESTDESC")
    _sub(reqdesc, "REPORTNAME", "Vouchers")
    statics = _sub(reqdesc, "STATICVARIABLES")
    _sub(statics, "SVCURRENTCOMPANY", company)
    reqdata = _sub(importdata, "REQUESTDATA")
    msg = etree.SubElement(reqdata, "TALLYMESSAGE", nsmap={"UDF": "TallyUDF"})

    v = etree.SubElement(
        msg, "VOUCHER", VCHTYPE="Purchase", ACTION="Create", OBJVIEW="Invoice Voucher View"
    )
    _sub(v, "DATE", vdate)
    _sub(v, "EFFECTIVEDATE", vdate)
    _sub(v, "VOUCHERTYPENAME", "Purchase")
    _sub(v, "VOUCHERNUMBER", invoice.invoice_number)
    _sub(v, "REFERENCE", invoice.invoice_number)
    _sub(v, "REFERENCEDATE", vdate)
    _sub(v, "PARTYLEDGERNAME", party_name)
    _sub(v, "PARTYNAME", party_name)
    _sub(v, "PARTYGSTIN", invoice.supplier.gstin)
    _sub(v, "GSTREGISTRATIONTYPE", "Regular")
    _sub(v, "STATENAME", _state_name(invoice.supplier.state_code))
    buyer_state = invoice.buyer.state_code or cfg.COMPANY_STATE_CODE
    _sub(v, "PLACEOFSUPPLY", _state_name(buyer_state, invoice.place_of_supply))
    _sub(v, "COUNTRYOFRESIDENCE", "India")
    _sub(v, "PERSISTEDVIEW", "Invoice Voucher View")
    _sub(v, "ISINVOICE", "Yes")
    full_narration = f"Auto-ingested GST invoice {invoice.invoice_number}"
    if invoice.po_number:
        full_narration += f" | PO {invoice.po_number}"
    if narration:
        full_narration += f" | {narration}"
    _sub(v, "NARRATION", full_narration[:900])

    # Party ledger — CREDIT, positive amount
    party = etree.SubElement(v, "LEDGERENTRIES.LIST")
    _sub(party, "LEDGERNAME", party_name)
    _sub(party, "ISDEEMEDPOSITIVE", "No")
    _sub(party, "ISPARTYLEDGER", "Yes")
    _sub(party, "ISLASTDEEMEDPOSITIVE", "No")
    _sub(party, "AMOUNT", _amt(party_amt))
    bill = etree.SubElement(party, "BILLALLOCATIONS.LIST")
    _sub(bill, "NAME", invoice.invoice_number)
    _sub(bill, "BILLTYPE", "New Ref")
    _sub(bill, "AMOUNT", _amt(party_amt))

    # Inventory + purchase allocation — DEBIT, negative amount
    for li, taxable in zip(invoice.line_items, line_taxables):
        inv_el = etree.SubElement(v, "ALLINVENTORYENTRIES.LIST")
        stock_name = li.matched_sku_name or li.description
        _sub(inv_el, "STOCKITEMNAME", stock_name)
        _sub(inv_el, "ISDEEMEDPOSITIVE", "Yes")
        _sub(inv_el, "ISLASTDEEMEDPOSITIVE", "Yes")
        qty = _qty_str(li.quantity, li.unit)

        if li.discount_pct:
            rate = _d(li.unit_price)
            _sub(inv_el, "RATE", f"{rate}/{li.unit}")
            _sub(inv_el, "DISCOUNT", f"{li.discount_pct:.2f}")
        elif li.discount_amount:
            net_rate = (taxable / Decimal(str(li.quantity))).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
            _sub(inv_el, "RATE", f"{net_rate}/{li.unit}")
        else:
            _sub(inv_el, "RATE", f"{_d(li.unit_price)}/{li.unit}")

        _sub(inv_el, "AMOUNT", _amt(-taxable))
        _sub(inv_el, "ACTUALQTY", qty)
        _sub(inv_el, "BILLEDQTY", qty)

        batch = etree.SubElement(inv_el, "BATCHALLOCATIONS.LIST")
        _sub(batch, "GODOWNNAME", cfg.TALLY_GODOWN)
        _sub(batch, "BATCHNAME", "Primary Batch")
        _sub(batch, "AMOUNT", _amt(-taxable))
        _sub(batch, "ACTUALQTY", qty)
        _sub(batch, "BILLEDQTY", qty)

        alloc = etree.SubElement(inv_el, "ACCOUNTINGALLOCATIONS.LIST")
        _sub(alloc, "LEDGERNAME", cfg.TALLY_LEDGER_PURCHASE)
        _sub(alloc, "ISDEEMEDPOSITIVE", "Yes")
        _sub(alloc, "AMOUNT", _amt(-taxable))

    # Input tax ledgers — DEBIT, negative amount
    for ledger, signed in tax_entries:
        t = etree.SubElement(v, "LEDGERENTRIES.LIST")
        _sub(t, "LEDGERNAME", ledger)
        _sub(t, "ISDEEMEDPOSITIVE", "Yes")
        _sub(t, "ISLASTDEEMEDPOSITIVE", "Yes")
        _sub(t, "AMOUNT", _amt(signed))

    # Round-off ledger (only when the printed grand total carries one)
    if roundoff_signed != 0:
        r = etree.SubElement(v, "LEDGERENTRIES.LIST")
        _sub(r, "LEDGERNAME", roundoff_ledger)
        is_debit = roundoff_signed < 0
        _sub(r, "ISDEEMEDPOSITIVE", "Yes" if is_debit else "No")
        _sub(r, "ISLASTDEEMEDPOSITIVE", "Yes" if is_debit else "No")
        _sub(r, "AMOUNT", _amt(roundoff_signed))

    xml = etree.tostring(env, pretty_print=True, encoding="unicode")

    # Independent post-build check on the serialised text itself
    verify_voucher_balanced(xml)
    return xml


# ─────────────────────────────────────────────────────────────────────────────
# Independent verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_voucher_balanced(xml: str) -> Decimal:
    """
    Re-parse the XML and prove every voucher balances to 0.00 with correct signs.
    Returns the (zero) net on success; raises TallyXMLError otherwise.
    """
    try:
        root = etree.fromstring(xml.encode("utf-8"))
    except etree.XMLSyntaxError as exc:
        raise TallyXMLError(f"Generated XML is not well-formed: {exc}") from exc

    net = _ZERO
    vouchers = root.findall(".//VOUCHER")
    if not vouchers:
        raise TallyXMLError("No <VOUCHER> element found.")

    for voucher in vouchers:
        net = _ZERO
        for entry in voucher.findall("LEDGERENTRIES.LIST"):
            amount = Decimal(entry.findtext("AMOUNT") or "0")
            positive = (entry.findtext("ISDEEMEDPOSITIVE") or "").strip()
            if positive == "Yes" and amount > 0:
                raise TallyXMLError(f"Sign error: debit entry with positive amount ({entry.findtext('LEDGERNAME')}).")
            if positive == "No" and amount < 0:
                raise TallyXMLError(f"Sign error: credit entry with negative amount ({entry.findtext('LEDGERNAME')}).")
            net += amount
        for inv_el in voucher.findall("ALLINVENTORYENTRIES.LIST"):
            inv_amount = Decimal(inv_el.findtext("AMOUNT") or "0")
            if (inv_el.findtext("ISDEEMEDPOSITIVE") or "") == "Yes" and inv_amount > 0:
                raise TallyXMLError("Sign error: inventory debit with positive amount.")
            alloc_sum = _ZERO
            for alloc in inv_el.findall("ACCOUNTINGALLOCATIONS.LIST"):
                alloc_sum += Decimal(alloc.findtext("AMOUNT") or "0")
            if alloc_sum != inv_amount:
                raise TallyXMLError(
                    f"Inventory amount {inv_amount} ≠ purchase allocation {alloc_sum} "
                    f"({inv_el.findtext('STOCKITEMNAME')})."
                )
            net += alloc_sum
        if round(net, 2) != 0:
            raise TallyXMLError(f"Voucher does not balance: net = {net}")
    return net


# ─────────────────────────────────────────────────────────────────────────────
# HTTP push
# ─────────────────────────────────────────────────────────────────────────────

def _decode_tally_response(content: bytes) -> str:
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return content.decode(enc)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")


def _parse_tally_response(text: str) -> dict[str, str]:
    """Pull CREATED / ALTERED / ERRORS / LASTVCHID / LINEERROR out of Tally's reply."""
    out = {"created": "0", "altered": "0", "errors": "0", "lastvchid": "", "lineerror": "", "exceptions": "0"}
    try:
        parser = etree.XMLParser(recover=True)
        root = etree.fromstring(text.strip().encode("utf-8"), parser=parser)
    except Exception:
        root = None
    if root is None:
        return out
    for key, tag in (
        ("created", "CREATED"), ("altered", "ALTERED"), ("errors", "ERRORS"),
        ("lastvchid", "LASTVCHID"), ("exceptions", "EXCEPTIONS"),
    ):
        found = root.find(f".//{tag}")
        if found is not None and found.text:
            out[key] = found.text.strip()
    line_errors = [e.text.strip() for e in root.iter("LINEERROR") if e.text]
    out["lineerror"] = " | ".join(line_errors)
    return out


def _voucher_number_from_xml(xml: str) -> str:
    try:
        root = etree.fromstring(xml.encode("utf-8"))
        return root.findtext(".//VOUCHERNUMBER") or ""
    except Exception:
        return ""


def push_to_tally(
    xml: str,
    *,
    invoice_id: str = "",
    host: str | None = None,
    port: int | None = None,
    timeout: int | None = None,
) -> TallyPushResult:
    """POST the voucher XML to TallyPrime's HTTP server (default http://localhost:9000)."""
    url = f"http://{host or cfg.TALLY_HOST}:{port or cfg.TALLY_PORT}"
    result = TallyPushResult(
        invoice_id=invoice_id,
        pushed_at=datetime.now().isoformat(timespec="seconds"),
        xml_payload_preview=xml[:500],
    )

    try:
        verify_voucher_balanced(xml)          # last line of defence
    except TallyXMLError as exc:
        result.error_message = f"Refused to push — {exc}"
        return result

    try:
        resp = httpx.post(
            url,
            content=xml.encode("utf-8"),
            headers={"Content-Type": "text/xml; charset=utf-8"},
            timeout=timeout or cfg.TALLY_TIMEOUT_SECONDS,
        )
    except httpx.ConnectError:
        result.error_message = (
            f"Cannot connect to TallyPrime at {url}. Open the company in TallyPrime and enable "
            "F1 › Settings › Connectivity › Client/Server (ODBC/HTTP) on this port."
        )
        return result
    except httpx.TimeoutException:
        result.error_message = f"Timed out after {timeout or cfg.TALLY_TIMEOUT_SECONDS}s waiting for Tally."
        return result
    except httpx.HTTPError as exc:
        result.error_message = f"HTTP error talking to Tally: {exc}"
        return result

    text = _decode_tally_response(resp.content)
    parsed = _parse_tally_response(text)
    result.http_status_code = resp.status_code
    result.tally_response_xml = text

    created, altered = int(parsed["created"] or 0), int(parsed["altered"] or 0)
    errors, exceptions = int(parsed["errors"] or 0), int(parsed["exceptions"] or 0)

    if resp.status_code == 200 and (created + altered) > 0 and errors == 0 and exceptions == 0:
        result.success = True
        result.tally_voucher_number = parsed["lastvchid"] or _voucher_number_from_xml(xml)
    else:
        result.error_message = (
            parsed["lineerror"]
            or f"Tally did not create the voucher (CREATED={created}, ERRORS={errors}, EXCEPTIONS={exceptions}). "
               "Check that the party ledger, stock items, godown and tax ledgers exist with exactly these names."
        )
    return result


def simulate_tally_push(xml: str, *, invoice_id: str = "") -> TallyPushResult:
    """Offline DEMO push: validates the XML exactly like a live push, but sends nothing."""
    result = TallyPushResult(
        invoice_id=invoice_id,
        pushed_at=datetime.now().isoformat(timespec="seconds"),
        xml_payload_preview=xml[:500],
        http_status_code=200,
    )
    try:
        verify_voucher_balanced(xml)
    except TallyXMLError as exc:
        result.error_message = f"Simulation rejected the voucher — {exc}"
        return result

    sim_id = f"SIM-{uuid.uuid4().hex[:6].upper()}"
    result.success = True
    result.tally_voucher_number = sim_id
    result.tally_response_xml = (
        "<RESPONSE><CREATED>1</CREATED><ALTERED>0</ALTERED><DELETED>0</DELETED>"
        f"<LASTVCHID>{sim_id}</LASTVCHID><ERRORS>0</ERRORS><EXCEPTIONS>0</EXCEPTIONS></RESPONSE>"
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from mock_data import SKU_MASTER, PO_MASTER, PARTY_MASTER, generate_mock_invoice
    from validator import validate_invoice

    raw = generate_mock_invoice(PARTY_MASTER, SKU_MASTER, PO_MASTER, seed=42, inject_errors=False)
    res = validate_invoice(raw, SKU_MASTER, PO_MASTER)
    inv = res.corrected_invoice
    assert inv is not None
    xml_out = build_purchase_voucher_xml(inv, narration="self-test")
    print(xml_out[:1800])
    print("…")
    print("Net of voucher:", verify_voucher_balanced(xml_out))
    print(simulate_tally_push(xml_out, invoice_id=inv.invoice_id).model_dump(exclude={"tally_response_xml", "xml_payload_preview"}))
