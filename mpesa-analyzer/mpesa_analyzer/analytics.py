"""Pandas aggregations that feed the dashboard."""
from __future__ import annotations

import pandas as pd

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"])
    out["date"] = ts.dt.date
    out["month"] = ts.dt.to_period("M").astype(str)
    out["weekday"] = ts.dt.dayofweek.map(dict(enumerate(WEEKDAYS)))
    out["hour"] = ts.dt.hour
    return out


def _spend(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["direction"] == "out") & (df["type"] != "fee")]


def kpis(df: pd.DataFrame) -> dict:
    inflow = float(df.loc[df["direction"] == "in", "amount"].sum())
    spend_df = _spend(df)
    spend = float(spend_df["amount"].sum())
    fees = float(df["fee"].fillna(0).sum() + df.loc[df["type"] == "fee", "amount"].sum())
    by_cat = spend_df.groupby("category")["amount"].sum().sort_values(ascending=False) if "category" in df else pd.Series(dtype=float)
    return {
        "inflow": inflow, "spend": spend, "fees": fees, "net": inflow - spend - fees,
        "count": int(len(df)),
        "top_category": by_cat.index[0] if len(by_cat) else "—",
        "top_category_amount": float(by_cat.iloc[0]) if len(by_cat) else 0.0,
        "fee_share": (fees / spend * 100) if spend else 0.0,
    }


def spend_by_category(df: pd.DataFrame) -> pd.DataFrame:
    g = _spend(df).groupby("category", as_index=False)["amount"].sum().sort_values("amount", ascending=False)
    return g.reset_index(drop=True)


def monthly_category(df: pd.DataFrame) -> pd.DataFrame:
    return _spend(df).groupby(["month", "category"], as_index=False)["amount"].sum()


def daily_flow(df: pd.DataFrame) -> pd.DataFrame:
    g = df[df["type"] != "fee"].groupby(["date", "direction"], as_index=False)["amount"].sum()
    return g


def top_counterparties(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    s = _spend(df)
    s = s[s["party"].astype(str).str.strip() != ""]
    g = s.groupby("party").agg(total=("amount", "sum"), transactions=("amount", "size"),
                               category=("category", "first")).reset_index()
    return g.sort_values("total", ascending=False).head(n).reset_index(drop=True)


def weekday_hour(df: pd.DataFrame) -> pd.DataFrame:
    s = _spend(df)
    m = s.pivot_table(index="weekday", columns="hour", values="amount", aggfunc="size", fill_value=0)
    return m.reindex(index=WEEKDAYS, columns=list(range(24)), fill_value=0)


def fees_by_month(df: pd.DataFrame) -> pd.DataFrame:
    f = df.assign(_fee=df["fee"].fillna(0) + df["amount"].where(df["type"] == "fee", 0))
    return f.groupby("month", as_index=False)["_fee"].sum().rename(columns={"_fee": "fees"})


def balance_series(df: pd.DataFrame) -> pd.DataFrame:
    b = df.dropna(subset=["balance"]).sort_values("timestamp")
    return b[["timestamp", "balance"]]


def recurring_payments(df: pd.DataFrame, min_count: int = 3) -> pd.DataFrame:
    s = _spend(df)
    s = s[(s["party"].astype(str).str.strip() != "") & (~s["type"].isin(["withdraw", "airtime"]))]
    g = s.groupby("party").agg(count=("amount", "size"), avg_amount=("amount", "mean"),
                               total=("amount", "sum"), first=("timestamp", "min"), last=("timestamp", "max"))
    g = g[g["count"] >= min_count].copy()
    g["avg_gap_days"] = (g["last"] - g["first"]).dt.total_seconds() / 86400 / (g["count"] - 1)
    return g.drop(columns=["first", "last"]).sort_values("total", ascending=False).reset_index()
