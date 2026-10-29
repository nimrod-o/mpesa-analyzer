"""Synthetic M-Pesa SMS generator. No real personal data is used anywhere."""
from __future__ import annotations

import calendar
import random
import string
from datetime import date, datetime, timedelta

import pandas as pd

# category -> list of (merchant name, kind, fixed paybill account or None)
MERCHANTS: dict[str, list[tuple[str, str, str | None]]] = {
    "Groceries": [
        ("NAIVAS SUPERMARKET", "buygoods", None), ("QUICKMART LTD", "buygoods", None),
        ("CARREFOUR KAREN", "buygoods", None), ("CHANDARANA FOODPLUS", "buygoods", None),
        ("FRESH MART KAWANGWARE", "buygoods", None), ("MAMA MBOGA FRESH", "buygoods", None),
        ("GREEN GROCER ROYSAMBU", "buygoods", None),
    ],
    "Transport": [
        ("UBER TRIP", "paybill", "UBER"), ("BOLT RIDES", "paybill", "BOLT"),
        ("SHELL KAREN", "buygoods", None), ("TOTAL ENERGIES THIKA RD", "buygoods", None),
        ("CHAPCHAP ENERGY", "buygoods", None), ("RUBIS ENERGY", "buygoods", None),
        ("LITTLE RIDE", "paybill", "LITTLE"),
    ],
    "Food & Dining": [
        ("KFC WESTLANDS", "buygoods", None), ("JAVA HOUSE", "buygoods", None),
        ("CHICKEN INN", "buygoods", None), ("ARTCAFFE", "buygoods", None),
        ("MAMA OLIECH HOTEL", "buygoods", None), ("SWEET TASTE EATERY", "buygoods", None),
    ],
    "Entertainment": [
        ("DSTV KENYA", "paybill", None), ("SHOWMAX", "paybill", None),
        ("NETFLIX", "paybill", None), ("CINEMAX THE HUB", "buygoods", None),
    ],
    "Betting": [
        ("SPORTPESA", "paybill", "888880"), ("BETIKA", "paybill", "888888"), ("ODIBETS", "paybill", None),
    ],
    "Health": [
        ("GOODLIFE PHARMACY", "buygoods", None), ("MOUNTAIN VIEW CLINIC", "buygoods", None),
        ("NAIROBI HOSPITAL", "paybill", None), ("HEALTHY U CHEMIST", "buygoods", None),
    ],
    "Education": [
        ("HELB", "paybill", None), ("SUNRISE ACADEMY", "paybill", None),
        ("KENYATTA UNIVERSITY", "paybill", None),
    ],
    "Savings & Investments": [
        ("KCB M-PESA", "paybill", None), ("UKULIMA SACCO", "paybill", None), ("M-SHWARI", "paybill", None),
    ],
    "Utilities": [
        ("KPLC PREPAID", "paybill", None), ("NAIROBI WATER", "paybill", None), ("ZUKU FIBER", "paybill", None),
    ],
}
PEOPLE = [
    "JOHN KAMAU", "JANE WANJIKU", "PETER OMONDI", "MARY ACHIENG", "DAVID KIPROP",
    "GRACE NJERI", "SAMUEL MWANGI", "FAITH AKINYI", "BRIAN OTIENO", "LUCY WAMBUI",
]
AGENTS = ["123456 - MAMA JOY AGENT NAIROBI", "234567 - TOTO FINANCIAL SERVICES", "345678 - KIOSK AGENT RONGAI"]
EMPLOYER = ("ACME HOLDINGS LTD", "0700112233")
LANDLORD = ("HOMELAND PROPERTIES", "A12 RENT")

AMOUNT_RANGES = {
    "Groceries": (150, 4500), "Transport": (100, 3500), "Food & Dining": (200, 2500),
    "Entertainment": (300, 3000), "Betting": (50, 2000), "Health": (200, 6000),
    "Education": (1000, 25000), "Savings & Investments": (500, 10000), "Utilities": (300, 3500),
}

_SENT_FEES = [(100, 0), (500, 7), (1000, 13), (1500, 23), (2500, 33), (3500, 53),
              (5000, 57), (7500, 78), (10000, 90), (15000, 100), (250000, 108)]
_WITHDRAW_FEES = [(100, 11), (500, 29), (1000, 29), (1500, 29), (2500, 52), (3500, 69),
                  (5000, 87), (7500, 115), (10000, 167), (250000, 185)]


def _fee(kind: str, amount: float) -> float:
    """Illustrative tariff only (not Safaricom's current rate card)."""
    table = {"sent": _SENT_FEES, "withdraw": _WITHDRAW_FEES}.get(kind)
    if not table:
        return 0.0
    for limit, fee in table:
        if amount <= limit:
            return float(fee)
    return float(table[-1][1])


def _money(x: float) -> str:
    return f"Ksh{x:,.2f}"


def _fmt_dt(dt: datetime) -> str:
    return f"{dt.day}/{dt.month}/{dt:%y} at {dt.hour % 12 or 12}:{dt.minute:02d} {'AM' if dt.hour < 12 else 'PM'}"


def _month_start(start: date, i: int) -> date:
    y = start.year + (start.month - 1 + i) // 12
    m = (start.month - 1 + i) % 12 + 1
    return date(y, m, 1)


def _render(kind: str, code: str, dt: datetime, amount: float, party: str, account: str,
            phone: str, balance: float, fee: float) -> str:
    a, b, f, when = _money(amount), _money(balance), _money(fee), _fmt_dt(dt)
    tail = f"New M-PESA balance is {b}. Transaction cost, {f}."
    if kind == "received":
        return f"{code} Confirmed. You have received {a} from {party} {phone} on {when}. {tail}"
    if kind == "sent":
        return f"{code} Confirmed. {a} sent to {party} {phone} on {when}. {tail}"
    if kind == "paybill":
        return f"{code} Confirmed. {a} sent to {party} for account {account} on {when}. {tail}"
    if kind == "buygoods":
        return f"{code} Confirmed. {a} paid to {party}. on {when}.{tail}"
    if kind == "airtime":
        return f"{code} Confirmed.You bought {a} of airtime on {when}.{tail}"
    if kind == "withdraw":
        return f"{code} Confirmed. on {when} Withdraw {a} from {party} {tail}"
    if kind == "deposit":
        return f"{code} Confirmed. On {when} Give {a} cash to {party} {tail}"
    raise ValueError(kind)


def generate(months: int = 3, seed: int = 42, start: date = date(2026, 1, 1),
             start_balance: float = 70000.0) -> pd.DataFrame:
    """Generate `months` of realistic synthetic activity as a DataFrame (incl. SMS text + true label)."""
    rng = random.Random(seed)
    phones = {p: f"07{rng.randint(10000000, 99999999)}" for p in PEOPLE}
    accounts: dict[str, str] = {}
    events: list[dict] = []

    def acct(name: str, fixed: str | None) -> str:
        if fixed:
            return fixed
        return accounts.setdefault(name, str(rng.randint(100000, 99999999)))

    def add(day_date: date, hour_range: tuple[int, int], kind: str, amount: float, party: str,
            account: str, phone: str, category: str) -> None:
        dt = datetime(day_date.year, day_date.month, day_date.day,
                      rng.randint(*hour_range), rng.randint(0, 59))
        events.append(dict(dt=dt, kind=kind, amount=float(amount), party=party,
                           account=account, phone=phone, category=category))

    def rnd_day(ms: date) -> date:
        return ms + timedelta(days=rng.randint(0, calendar.monthrange(ms.year, ms.month)[1] - 1))

    def merchant_event(ms: date, category: str, hours=(7, 21)) -> None:
        name, kind, fixed = rng.choice(MERCHANTS[category])
        lo, hi = AMOUNT_RANGES[category]
        amount = round(rng.randint(lo, hi) / 10) * 10
        add(rnd_day(ms), hours, kind, amount, name, acct(name, fixed) if kind == "paybill" else "", "", category)

    for i in range(months):
        ms = _month_start(start, i)
        dim = calendar.monthrange(ms.year, ms.month)[1]
        add(date(ms.year, ms.month, min(28, dim)), (8, 9), "received", rng.choice(range(45000, 90001, 500)),
            EMPLOYER[0], "", EMPLOYER[1], "Income")
        add(ms + timedelta(days=rng.randint(0, 4)), (8, 11), "paybill", 15000.0, LANDLORD[0], LANDLORD[1], "", "Rent")
        for cat, n in [("Groceries", 6), ("Transport", 14), ("Food & Dining", 8), ("Utilities", 4),
                       ("Entertainment", 2), ("Savings & Investments", 2), ("Betting", rng.randint(0, 5)),
                       ("Health", rng.randint(0, 2)), ("Education", rng.randint(0, 1))]:
            for _ in range(n):
                merchant_event(ms, cat, (17, 22) if cat in ("Betting", "Entertainment") else (7, 21))
        for _ in range(4):
            add(rnd_day(ms), (7, 20), "airtime", rng.choice([20, 50, 100, 200, 500]), "", "", "", "Airtime & Data")
        for _ in range(5):
            p = rng.choice(PEOPLE)
            add(rnd_day(ms), (8, 21), "sent", rng.choice([100, 200, 300, 500, 1000, 1500, 2000, 3000]),
                p, "", phones[p], "Person-to-Person")
        for _ in range(3):
            p = rng.choice(PEOPLE)
            add(rnd_day(ms), (8, 21), "received", rng.choice([200, 500, 1000, 2500, 4000]), p, "", phones[p], "Income")
        for _ in range(3):
            add(rnd_day(ms), (9, 19), "withdraw", rng.choice([500, 1000, 2000, 3000, 5000]),
                rng.choice(AGENTS), "", "", "Cash Withdrawal")
        add(rnd_day(ms), (9, 19), "deposit", rng.choice([1000, 2000, 5000]), rng.choice(AGENTS).split(" - ")[1],
            "", "", "Cash Deposit")

    events.sort(key=lambda e: e["dt"])
    balance, used, rows = float(start_balance), set(), []
    for e in events:
        fee = _fee(e["kind"], e["amount"])
        inflow = e["kind"] in ("received", "deposit")
        if not inflow and e["amount"] + fee > balance:
            continue  # insufficient funds: skip
        balance += e["amount"] if inflow else -(e["amount"] + fee)
        code = "".join(rng.choices(string.ascii_uppercase + string.digits, k=10))
        while code in used:
            code = "".join(rng.choices(string.ascii_uppercase + string.digits, k=10))
        used.add(code)
        sms = _render(e["kind"], code, e["dt"], e["amount"], e["party"], e["account"], e["phone"], balance, fee)
        rows.append(dict(code=code, timestamp=e["dt"], type=e["kind"], direction="in" if inflow else "out",
                         amount=e["amount"], party=e["party"], account=e["account"], phone=e["phone"],
                         balance=round(balance, 2), fee=fee, true_category=e["category"], sms=sms))
    return pd.DataFrame(rows)


def sms_text(df: pd.DataFrame) -> str:
    """Join generated messages into a pasteable block of SMS text."""
    return "\n\n".join(df["sms"])
