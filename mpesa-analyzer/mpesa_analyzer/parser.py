"""Parse M-Pesa SMS text and CSV statement exports into a clean DataFrame."""
from __future__ import annotations

import re
from datetime import datetime

import pandas as pd

from .config import COLUMNS

_CODE = r"(?P<code>(?-i:[A-Z0-9]{10}))"
_CONF = r"\s+(?i:Confirmed)\.?\s*"
_AMT = r"Ksh\s?(?P<amount>[\d,]+(?:\.\d{2})?)"
_DT = r"(?P<date>\d{1,2}/\d{1,2}/\d{2,4}) at (?P<time>\d{1,2}:\d{2}\s?[AaPp][Mm])"
_PHONE = r"(?P<phone>[\d*+]{9,13})"

# Order matters: paybill must be tried before person-to-person "sent".
PATTERNS: list[tuple[str, re.Pattern]] = [
    ("received", re.compile(rf"{_CODE}{_CONF}You have received {_AMT} from (?P<party>.+?)\s+{_PHONE}\s+on {_DT}", re.I)),
    ("paybill", re.compile(rf"{_CODE}{_CONF}{_AMT} sent to (?P<party>.+?) for account (?P<account>.+?) on {_DT}", re.I)),
    ("sent", re.compile(rf"{_CODE}{_CONF}{_AMT} sent to (?P<party>.+?)\s+{_PHONE}\s+on {_DT}", re.I)),
    ("buygoods", re.compile(rf"{_CODE}{_CONF}{_AMT} paid to (?P<party>.+?)\.?\s+on {_DT}", re.I)),
    ("airtime", re.compile(rf"{_CODE}{_CONF}You bought {_AMT} of airtime on {_DT}", re.I)),
    ("withdraw", re.compile(rf"{_CODE}{_CONF}on {_DT}\s+Withdraw {_AMT} from (?P<party>.+?)\s+New M-PESA balance", re.I)),
    ("deposit", re.compile(rf"{_CODE}{_CONF}on {_DT}\s+Give {_AMT} cash to (?P<party>.+?)\s+New M-PESA balance", re.I)),
]
_BALANCE = re.compile(r"New M-PESA balance is Ksh\s?(?P<v>[\d,]+(?:\.\d{2})?)", re.I)
_FEE = re.compile(r"Transaction cost,?\s*Ksh\s?(?P<v>[\d,]+(?:\.\d{2})?)", re.I)
_SPLIT = re.compile(r"(?=(?<![A-Za-z0-9])[A-Z0-9]{10}\s+(?i:Confirmed))")

INFLOW_TYPES = {"received", "deposit"}


def _to_float(s: str | None) -> float:
    if s is None or str(s).strip() == "":
        return float("nan")
    return float(str(s).replace(",", "").strip())


def _parse_dt(date_s: str, time_s: str) -> datetime:
    time_s = re.sub(r"\s+", " ", time_s.strip().upper())
    if " " not in time_s:
        time_s = time_s[:-2] + " " + time_s[-2:]
    year_fmt = "%Y" if len(date_s.split("/")[-1]) == 4 else "%y"
    return datetime.strptime(f"{date_s} {time_s}", f"%d/%m/{year_fmt} %I:%M %p")


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS).astype(
        {"timestamp": "datetime64[ns]", "amount": float, "balance": float, "fee": float}
    )


def parse_sms(text: str) -> dict | None:
    """Parse one M-Pesa SMS. Returns a record dict, or None if unrecognised."""
    text = " ".join(str(text).split())
    for kind, pattern in PATTERNS:
        m = pattern.search(text)
        if not m:
            continue
        g = m.groupdict()
        bal, fee = _BALANCE.search(text), _FEE.search(text)
        return {
            "code": g["code"].upper(),
            "timestamp": _parse_dt(g["date"], g["time"]),
            "type": kind,
            "direction": "in" if kind in INFLOW_TYPES else "out",
            "amount": _to_float(g["amount"]),
            "party": (g.get("party") or "").strip(" ."),
            "account": (g.get("account") or "").strip(),
            "phone": g.get("phone") or "",
            "balance": _to_float(bal.group("v")) if bal else float("nan"),
            "fee": _to_float(fee.group("v")) if fee else 0.0,
            "raw": text,
        }
    return None


def split_messages(blob: str) -> list[str]:
    """Split a pasted block of SMS text into individual messages."""
    return [c.strip() for c in _SPLIT.split(blob or "") if c.strip()]


def parse_text(blob: str) -> tuple[pd.DataFrame, list[str]]:
    """Parse a blob of many SMS messages. Returns (transactions, unrecognised_chunks)."""
    records, unparsed = [], []
    for chunk in split_messages(blob):
        try:
            rec = parse_sms(chunk)
        except ValueError:  # e.g. impossible date
            rec = None
        if rec:
            records.append(rec)
        else:
            unparsed.append(chunk)
    if not records:
        return empty_frame(), unparsed
    df = pd.DataFrame(records, columns=COLUMNS)
    df = df.drop_duplicates("code").sort_values("timestamp", kind="stable").reset_index(drop=True)
    return df, unparsed


# ----------------------------------------------------------------- statement CSV
_STMT_RULES = [
    ("received", re.compile(r"^Funds received from\s*-?\s*(?P<party>.+)$", re.I)),
    ("paybill", re.compile(r"^Pay ?Bill(?: Online)? to (?:\d+)\s*-\s*(?P<party>.+?)\s+Acc\.?\s*(?P<account>.+)$", re.I)),
    ("buygoods", re.compile(r"^(?:Merchant Payment|Buy Goods)(?: Online)? to (?:\d+)\s*-\s*(?P<party>.+)$", re.I)),
    ("sent", re.compile(r"^Customer Transfer(?: to)?\s*-?\s*(?P<party>.+)$", re.I)),
    ("airtime", re.compile(r"^Airtime Purchase(?P<party>.*)$", re.I)),
    ("withdraw", re.compile(r"^Customer Withdrawal(?: At Agent)?\s*-?\s*(?P<party>.*)$", re.I)),
    ("deposit", re.compile(r"^(?:Deposit of Funds at Agent|Cash Deposit)\s*-?\s*(?P<party>.*)$", re.I)),
]


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(r"[^\d.\-]", "", regex=True), errors="coerce")


def parse_statement(source) -> pd.DataFrame:
    """Parse an M-Pesa statement exported as CSV (path or file-like object)."""
    raw = pd.read_csv(source, dtype=str, keep_default_na=False)
    raw.columns = [re.sub(r"[^a-z]", "", str(c).lower()) for c in raw.columns]
    for needed in ("completiontime", "details"):
        if needed not in raw.columns:
            raise ValueError("Not an M-Pesa statement CSV: missing 'Completion Time' / 'Details' columns.")

    n = len(raw)
    zeros = pd.Series([0.0] * n, index=raw.index)
    paid_in = _num(raw["paidin"]).fillna(0.0) if "paidin" in raw.columns else zeros
    withdrawn = _num(raw["withdrawn"]).fillna(0.0).abs() if "withdrawn" in raw.columns else zeros
    balance = _num(raw["balance"]) if "balance" in raw.columns else pd.Series([float("nan")] * n, index=raw.index)
    when = pd.to_datetime(raw["completiontime"], errors="coerce")

    records = []
    for i in raw.index:
        if pd.isna(when[i]):
            continue
        status = raw["transactionstatus"][i].strip().lower() if "transactionstatus" in raw.columns else ""
        if status and status != "completed":
            continue
        details = " ".join(raw["details"][i].split())
        kind, party, account = "other", details, ""
        if re.search(r"charge", details, re.I):
            kind = "fee"
        else:
            for k, pat in _STMT_RULES:
                m = pat.match(details)
                if m:
                    kind = k
                    party = (m.groupdict().get("party") or "").strip(" -")
                    account = (m.groupdict().get("account") or "").strip()
                    break
        is_in = paid_in[i] > 0
        amount = paid_in[i] if is_in else withdrawn[i]
        if amount <= 0:
            continue
        records.append({
            "code": raw["receiptno"][i].strip() if "receiptno" in raw.columns else f"ROW{i}",
            "timestamp": when[i].to_pydatetime(),
            "type": kind,
            "direction": "in" if is_in else "out",
            "amount": float(amount),
            "party": party,
            "account": account,
            "phone": "",
            "balance": float(balance[i]) if not pd.isna(balance[i]) else float("nan"),
            "fee": 0.0,
            "raw": details,
        })
    if not records:
        return empty_frame()
    return pd.DataFrame(records, columns=COLUMNS).sort_values("timestamp", kind="stable").reset_index(drop=True)
