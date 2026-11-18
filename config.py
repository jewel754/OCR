"""
config.py — Central configuration hub for the MSME GST Invoice Engine.

Usage:
    from config import cfg

    if cfg.DEMO_MODE:
        # use synthetic data + simulated VLM responses
    else:
        # call real Gemini / OpenAI / Anthropic Vision APIs

All settings can be overridden via environment variables or a local .env file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env from project root (sibling to this file)
load_dotenv(Path(__file__).parent / ".env", override=False)


# ─────────────────────────────────────────────────────────────────────────────
# Settings model
# ─────────────────────────────────────────────────────────────────────────────

class AppSettings(BaseSettings):
    """
    All tuneable knobs for the pipeline.

    Priority order (highest → lowest):
      1. Environment variables
      2. .env file
      3. Defaults defined here
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Master toggle ──────────────────────────────────────────────────────
    # Set DEMO_MODE=False in .env (or env var) to enable live VLM API calls.
    # When True, every pipeline stage uses deterministic synthetic data so
    # the full app runs with ZERO paid API keys.
    DEMO_MODE: bool = True

    # ── Company / GST identity ─────────────────────────────────────────────
    COMPANY_NAME: str = "Acme Trading Pvt Ltd"
    COMPANY_GSTIN: str = "27AABCU9603R1ZX"   # Maharashtra sample GSTIN
    COMPANY_STATE_CODE: str = "27"            # Maharashtra

    # ── VLM provider selection ─────────────────────────────────────────────
    # Choices: "gemini" | "openai" | "anthropic"
    VLM_PROVIDER: Literal["gemini", "openai", "anthropic"] = "gemini"

    # API keys — read from env; empty string = not configured
    GEMINI_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""

    # Model names (override via env if newer models release)
    GEMINI_MODEL: str = "gemini-1.5-flash"
    OPENAI_MODEL: str = "gpt-4o"
    ANTHROPIC_MODEL: str = "claude-sonnet-4-6"

    # ── DoubleTake loop settings ───────────────────────────────────────────
    # DoubleTake = when page-bottom row is likely cropped, the engine
    # re-crops the PDF/image with an N-pixel vertical overlap and re-runs
    # VLM extraction on that sliver only, then merges the extra rows.
    DOUBLETAKE_OVERLAP_PX: int = 120    # extra pixels to include on re-crop
    DOUBLETAKE_MAX_RETRIES: int = 2     # maximum additional extraction passes
    DOUBLETAKE_TRIGGER_THRESHOLD: float = 0.85  # line-item total / invoice total; if ratio < this, trigger

    # ── Validation thresholds ──────────────────────────────────────────────
    # Maximum permissible rounding error (₹) on individual line tax totals
    TAX_ROUNDING_TOLERANCE: float = 1.00

    # Maximum permissible rounding error (₹) on grand total
    GRAND_TOTAL_TOLERANCE: float = 2.00

    # RapidFuzz fuzzy-match minimum score (0–100) for SKU reconciliation
    SKU_MATCH_MIN_SCORE: float = 72.0

    # Party-name fuzzy match threshold
    PARTY_MATCH_MIN_SCORE: float = 65.0

    # Rate deviation: flag if extracted unit price deviates from PO by > X%
    PO_RATE_DEVIATION_PCT: float = 5.0

    # ── OCR / Tesseract ────────────────────────────────────────────────────
    # Path to tesseract binary; leave empty to use system PATH
    TESSERACT_CMD: str = ""

    # Tesseract config string for invoice text
    TESSERACT_CONFIG: str = "--oem 3 --psm 6 -l eng+hin"

    # ── DEMO error-injection settings ─────────────────────────────────────
    # In DEMO_MODE, the mock extractor intentionally injects errors so the
    # validator's correction logic can be demonstrated end-to-end.
    DEMO_INJECT_MATH_ERROR: bool = True       # one line subtotal off by ₹1
    DEMO_INJECT_OCR_ERROR: bool = True        # one GSTIN digit swapped
    DEMO_INJECT_MISSING_ROW: bool = True      # simulate a cropped last row

    # ── Tally integration ──────────────────────────────────────────────────
    TALLY_HOST: str = "localhost"
    TALLY_PORT: int = 9000
    TALLY_COMPANY_NAME: str = "Acme Trading Pvt Ltd"
    TALLY_LEDGER_PURCHASE: str = "Purchase Account"
    TALLY_LEDGER_CGST: str = "CGST Input"
    TALLY_LEDGER_SGST: str = "SGST Input"
    TALLY_LEDGER_IGST: str = "IGST Input"
    TALLY_GODOWN: str = "Main Location"
    TALLY_TIMEOUT_SECONDS: int = 10

    # ── Paths ──────────────────────────────────────────────────────────────
    BASE_DIR: Path = Path(__file__).parent
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    OUTPUT_DIR: Path = BASE_DIR / "outputs"
    MASTER_DIR: Path = BASE_DIR / "masters"

    # ── Logging ───────────────────────────────────────────────────────────
    LOG_LEVEL: str = "DEBUG"
    LOG_FILE: str = "pipeline.log"

    @field_validator("COMPANY_GSTIN")
    @classmethod
    def validate_gstin_format(cls, v: str) -> str:
        """Basic GSTIN structural check: 15 alphanumeric chars."""
        v = v.strip().upper()
        if len(v) != 15:
            raise ValueError(f"GSTIN must be exactly 15 characters, got {len(v)}: {v!r}")
        return v

    @field_validator("COMPANY_STATE_CODE")
    @classmethod
    def validate_state_code(cls, v: str) -> str:
        valid_codes = {
            "01","02","03","04","05","06","07","08","09","10",
            "11","12","13","14","15","16","17","18","19","20",
            "21","22","23","24","25","26","27","28","29","30",
            "31","32","33","34","35","36","37","38","97","99",
        }
        if v not in valid_codes:
            raise ValueError(f"Unknown Indian state code: {v!r}")
        return v

    def tally_url(self) -> str:
        """Return the full Tally HTTP endpoint URL."""
        return f"http://{self.TALLY_HOST}:{self.TALLY_PORT}"

    def ensure_dirs(self) -> None:
        """Create runtime directories if they don't exist."""
        for d in (self.UPLOAD_DIR, self.OUTPUT_DIR, self.MASTER_DIR):
            d.mkdir(parents=True, exist_ok=True)

    def vlm_api_key(self) -> str:
        """Return the API key for the selected VLM provider."""
        mapping = {
            "gemini": self.GEMINI_API_KEY,
            "openai": self.OPENAI_API_KEY,
            "anthropic": self.ANTHROPIC_API_KEY,
        }
        return mapping[self.VLM_PROVIDER]

    def vlm_is_configured(self) -> bool:
        """Return True if the selected VLM provider has a non-empty API key."""
        return bool(self.vlm_api_key())


# ─────────────────────────────────────────────────────────────────────────────
# Singleton instance — import this everywhere
# ─────────────────────────────────────────────────────────────────────────────
cfg = AppSettings()
cfg.ensure_dirs()


# ─────────────────────────────────────────────────────────────────────────────
# Indian GST reference tables
# ─────────────────────────────────────────────────────────────────────────────

# State code → state name (GST registration states)
INDIAN_STATE_CODES: dict[str, str] = {
    "01": "Jammu & Kashmir",
    "02": "Himachal Pradesh",
    "03": "Punjab",
    "04": "Chandigarh",
    "05": "Uttarakhand",
    "06": "Haryana",
    "07": "Delhi",
    "08": "Rajasthan",
    "09": "Uttar Pradesh",
    "10": "Bihar",
    "11": "Sikkim",
    "12": "Arunachal Pradesh",
    "13": "Nagaland",
    "14": "Manipur",
    "15": "Mizoram",
    "16": "Tripura",
    "17": "Meghalaya",
    "18": "Assam",
    "19": "West Bengal",
    "20": "Jharkhand",
    "21": "Odisha",
    "22": "Chhattisgarh",
    "23": "Madhya Pradesh",
    "24": "Gujarat",
    "25": "Daman & Diu",
    "26": "Dadra & Nagar Haveli",
    "27": "Maharashtra",
    "28": "Andhra Pradesh",
    "29": "Karnataka",
    "30": "Goa",
    "31": "Lakshadweep",
    "32": "Kerala",
    "33": "Tamil Nadu",
    "34": "Puducherry",
    "35": "Andaman & Nicobar",
    "36": "Telangana",
    "37": "Andhra Pradesh (New)",
    "38": "Ladakh",
    "97": "Other Territory",
    "99": "Centre Jurisdiction",
}

# Valid GST rates (%) for Indian goods/services
VALID_GST_RATES: frozenset[float] = frozenset({0.0, 0.1, 0.25, 1.0, 1.5, 3.0, 5.0, 6.0, 7.5, 9.0, 12.0, 14.0, 18.0, 28.0})

# CGST = SGST = GST_RATE / 2  (intra-state)
# IGST = GST_RATE              (inter-state)

# SAC / HSN code length validation rules
HSN_MIN_DIGITS: int = 4   # goods
SAC_MIN_DIGITS: int = 4   # services

# GSTIN checksum alphabet
GSTIN_ALPHABET: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def gstin_state_code(gstin: str) -> str:
    """Extract the 2-digit state code from a GSTIN."""
    return gstin[:2]


def is_inter_state(supplier_gstin: str, buyer_gstin: str) -> bool:
    """
    Return True if the transaction is inter-state (IGST applies),
    False if intra-state (CGST + SGST apply).
    """
    return gstin_state_code(supplier_gstin) != gstin_state_code(buyer_gstin)


def compute_gstin_checksum(gstin_14: str) -> str:
    """
    Compute the 15th character (checksum) of a GSTIN using the
    official Government of India modulus-36 algorithm.

    Args:
        gstin_14: First 14 characters of a GSTIN.

    Returns:
        Single character checksum.
    """
    factor = 1
    total = 0
    code_point_count = len(GSTIN_ALPHABET)

    for char in gstin_14:
        digit = GSTIN_ALPHABET.index(char)
        addend = factor * digit
        addend = (addend // code_point_count) + (addend % code_point_count)
        total += addend
        factor = 1 if factor == 2 else 2

    remainder = total % code_point_count
    return GSTIN_ALPHABET[(code_point_count - remainder) % code_point_count]


def validate_gstin_checksum(gstin: str) -> bool:
    """Return True if the GSTIN passes the official checksum algorithm."""
    if len(gstin) != 15:
        return False
    expected = compute_gstin_checksum(gstin[:14])
    return gstin[14] == expected
