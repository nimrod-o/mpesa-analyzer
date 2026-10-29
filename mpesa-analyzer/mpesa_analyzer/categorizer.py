"""Two-stage categorization: transaction type + regex rules first, ML classifier as fallback."""
from __future__ import annotations

import re

import pandas as pd

from .classifier import ensure_model, predict, row_text

RULES: list[tuple[str, re.Pattern]] = [
    ("Rent", re.compile(r"\bRENT\b|PROPERTIES|ESTATES?\b|LANDLORD|REAL ?ESTATE", re.I)),
    ("Utilities", re.compile(r"KPLC|KENYA POWER|NAIROBI WATER|WATER (?:CO|SERVICES)|ZUKU|FIB(?:RE|ER)|WIFI|STIMA", re.I)),
    ("Groceries", re.compile(r"NAIVAS|QUICKMART|CARREFOUR|CHANDARANA|SUPERMARKET|TUSKYS|CLEANSHELF", re.I)),
    ("Transport", re.compile(r"UBER|BOLT|\bLITTLE\b|SHELL|TOTAL ?ENERG|RUBIS|PETROL|FUEL|\bSGR\b|MATATU", re.I)),
    ("Food & Dining", re.compile(r"KFC|JAVA|CHICKEN INN|ARTCAFFE|PIZZA|CAFE|RESTAURANT", re.I)),
    ("Entertainment", re.compile(r"DSTV|GOTV|SHOWMAX|NETFLIX|CINEMA|CINEMAX|SPOTIFY", re.I)),
    ("Betting", re.compile(r"SPORTPESA|BETIKA|ODIBETS|MOZZART|BETWAY|\bBET\b", re.I)),
    ("Health", re.compile(r"PHARMACY|HOSPITAL|CHEMIST|DENTAL", re.I)),
    ("Education", re.compile(r"HELB|UNIVERSITY|SCHOOL|COLLEGE|KNEC", re.I)),
    ("Savings & Investments", re.compile(r"M-?SHWARI|KCB M-?PESA|SACCO|\bMMF\b|\bETF\b|NCBA LOOP", re.I)),
]
TYPE_CATEGORY = {
    "received": "Income", "deposit": "Cash Deposit", "withdraw": "Cash Withdrawal",
    "airtime": "Airtime & Data", "sent": "Person-to-Person", "fee": "Fees & Charges",
}


def rule_category(kind: str, party: str, account: str = "") -> str | None:
    """Return a category from type/regex rules, or None if the rules cannot decide."""
    if kind in TYPE_CATEGORY:
        return TYPE_CATEGORY[kind]
    text = f"{party} {account}"
    for category, pattern in RULES:
        if pattern.search(text):
            return category
    return None


class Categorizer:
    def __init__(self, bundle: dict | None = None, threshold: float = 0.45):
        self.bundle = bundle if bundle is not None else ensure_model()
        self.threshold = threshold

    def categorize(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["category"], out["method"], out["confidence"] = "Other", "default", 0.0
        if out.empty:
            return out
        pending = []
        for i, r in enumerate(out.itertuples()):
            cat = rule_category(r.type, r.party, r.account)
            if cat:
                out.iloc[i, out.columns.get_loc("category")] = cat
                out.iloc[i, out.columns.get_loc("method")] = "rule"
                out.iloc[i, out.columns.get_loc("confidence")] = 1.0
            else:
                pending.append(i)
        if pending:
            texts = [row_text(out.iloc[i]["type"], out.iloc[i]["party"], out.iloc[i]["account"],
                              out.iloc[i]["amount"]) for i in pending]
            for i, (label, conf) in zip(pending, predict(self.bundle, texts)):
                out.iloc[i, out.columns.get_loc("confidence")] = conf
                if conf >= self.threshold:
                    out.iloc[i, out.columns.get_loc("category")] = label
                    out.iloc[i, out.columns.get_loc("method")] = "ml"
        return out

    def predict_one(self, kind: str, party: str, account: str = "", amount: float = 500.0) -> tuple[str, float, str]:
        cat = rule_category(kind, party, account)
        if cat:
            return cat, 1.0, "rule"
        label, conf = predict(self.bundle, [row_text(kind, party, account, amount)])[0]
        return (label, conf, "ml") if conf >= self.threshold else ("Other", conf, "default")
