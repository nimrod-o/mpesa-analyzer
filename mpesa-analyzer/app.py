"""M-Pesa Statement Analyzer — Streamlit dashboard.  Run:  streamlit run app.py"""
import io

import pandas as pd
import plotly.express as px
import streamlit as st

from mpesa_analyzer import analytics, pipeline
from mpesa_analyzer.categorizer import Categorizer
from mpesa_analyzer.config import CATEGORIES, DATA_DIR

st.set_page_config(page_title="M-Pesa Statement Analyzer", page_icon="💸", layout="wide")


@st.cache_resource(show_spinner="Preparing classifier…")
def get_categorizer() -> Categorizer:
    return Categorizer()


@st.cache_data(show_spinner="Generating synthetic data…")
def load_synthetic(months: int, seed: int):
    return pipeline.from_synthetic(months, seed, get_categorizer())


@st.cache_data(show_spinner="Parsing messages…")
def load_text(text: str):
    return pipeline.from_text(text, get_categorizer())


@st.cache_data(show_spinner="Parsing statement…")
def load_statement(data: bytes):
    return pipeline.from_statement(io.BytesIO(data), get_categorizer())


def ksh(x: float) -> str:
    return f"KSh {x:,.0f}"


# ------------------------------------------------------------------ sidebar
st.sidebar.title("💸 M-Pesa Analyzer")
source = st.sidebar.radio("Data source", ["Synthetic sample", "Paste SMS text", "Upload file"])
st.sidebar.caption("Everything is processed in memory. Nothing is stored or sent anywhere.")

df, unparsed = pd.DataFrame(), []
if source == "Synthetic sample":
    months = st.sidebar.slider("Months of data", 1, 12, 6)
    seed = st.sidebar.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)
    df, unparsed = load_synthetic(int(months), int(seed))
elif source == "Paste SMS text":
    text = st.sidebar.text_area("Paste M-Pesa SMS messages", height=250,
                                placeholder="QGH7A1B2C3 Confirmed. You have received Ksh1,500.00 from ...")
    if text.strip():
        df, unparsed = load_text(text)
else:
    up = st.sidebar.file_uploader("SMS export (.txt) or statement (.csv)", type=["txt", "csv"])
    if up is not None:
        data = up.getvalue()
        if up.name.lower().endswith(".csv"):
            try:
                df, unparsed = load_statement(data)
            except ValueError as err:
                st.sidebar.warning(f"{err} Trying to read it as SMS text instead.")
                df, unparsed = load_text(data.decode("utf-8", errors="replace"))
        else:
            df, unparsed = load_text(data.decode("utf-8", errors="replace"))

st.title("M-Pesa Statement Analyzer")
st.caption("Parse → categorize (regex rules + ML fallback) → explore your spending patterns.")

if unparsed:
    st.sidebar.warning(f"{len(unparsed)} message(s) were not recognised.")
    with st.sidebar.expander("Show unrecognised text"):
        for chunk in unparsed[:20]:
            st.code(chunk[:300])

if df.empty:
    st.info("Pick **Synthetic sample** in the sidebar for a demo, or paste / upload your own messages.")
    st.stop()

# ------------------------------------------------------------------ filters
dmin, dmax = df["timestamp"].min().date(), df["timestamp"].max().date()
picked = st.sidebar.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
if not isinstance(picked, (tuple, list)) or len(picked) != 2:
    st.info("Select an end date to continue.")
    st.stop()
cats = st.sidebar.multiselect("Categories", sorted(df["category"].unique()), default=sorted(df["category"].unique()))

mask = (df["timestamp"].dt.date >= picked[0]) & (df["timestamp"].dt.date <= picked[1]) & df["category"].isin(cats)
f = df[mask]
if f.empty:
    st.warning("No transactions match the current filters.")
    st.stop()

# ------------------------------------------------------------------ KPIs
k = analytics.kpis(f)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Money in", ksh(k["inflow"]))
c2.metric("Money out", ksh(k["spend"]))
c3.metric("Net flow", ksh(k["net"]))
c4.metric("Fees paid", ksh(k["fees"]), f"{k['fee_share']:.1f}% of spend", delta_color="off")
c5.metric("Top category", k["top_category"], ksh(k["top_category_amount"]), delta_color="off")

tab_over, tab_cat, tab_merch, tab_pat, tab_tx, tab_model = st.tabs(
    ["Overview", "Categories", "Merchants", "Patterns", "Transactions", "Classifier"])

with tab_over:
    flow = analytics.daily_flow(f)
    st.plotly_chart(px.bar(flow, x="date", y="amount", color="direction", barmode="group",
                           title="Daily money in vs out"), use_container_width=True)
    bal = analytics.balance_series(f)
    if not bal.empty:
        st.plotly_chart(px.line(bal, x="timestamp", y="balance", title="M-Pesa balance over time"),
                        use_container_width=True)

with tab_cat:
    left, right = st.columns(2)
    sc = analytics.spend_by_category(f)
    if sc.empty:
        st.info("No spending in the selected range.")
    else:
        left.plotly_chart(px.pie(sc, names="category", values="amount", hole=0.45, title="Share of spending"),
                          use_container_width=True)
        right.plotly_chart(px.bar(sc, x="amount", y="category", orientation="h", title="Spending by category")
                           .update_layout(yaxis={"categoryorder": "total ascending"}), use_container_width=True)
        st.plotly_chart(px.bar(analytics.monthly_category(f), x="month", y="amount", color="category",
                               title="Monthly spending by category"), use_container_width=True)

with tab_merch:
    top = analytics.top_counterparties(f, 10)
    if not top.empty:
        st.plotly_chart(px.bar(top, x="total", y="party", color="category", orientation="h",
                               title="Top 10 recipients by amount")
                        .update_layout(yaxis={"categoryorder": "total ascending"}), use_container_width=True)
    st.subheader("Recurring payments")
    rec = analytics.recurring_payments(f)
    if rec.empty:
        st.caption("No recipient was paid 3+ times in this range.")
    else:
        st.dataframe(rec.round(1), use_container_width=True, hide_index=True)

with tab_pat:
    hm = analytics.weekday_hour(f)
    st.plotly_chart(px.imshow(hm, aspect="auto", color_continuous_scale="Greens",
                              labels=dict(x="Hour of day", y="Weekday", color="Payments"),
                              title="When do you spend? (number of payments)"), use_container_width=True)
    fees = analytics.fees_by_month(f)
    st.plotly_chart(px.bar(fees, x="month", y="fees", title="Transaction fees per month"),
                    use_container_width=True)

with tab_tx:
    cols = ["timestamp", "type", "direction", "amount", "party", "account", "fee", "balance",
            "category", "method", "confidence"]
    view = f[cols].sort_values("timestamp", ascending=False)
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.download_button("Download CSV", view.to_csv(index=False).encode("utf-8"), "mpesa_transactions.csv", "text/csv")

with tab_model:
    cat = get_categorizer()
    counts = f["method"].value_counts().rename_axis("method").reset_index(name="transactions")
    st.write("**How each transaction was categorized**")
    st.dataframe(counts, hide_index=True)
    m = cat.bundle.get("metrics", {})
    if m:
        st.metric("Holdout accuracy (merchants never seen in training)", f"{m['holdout_accuracy']:.1%}")
        st.caption(f"Trained on {m['n_train']} synthetic payments, tested on {m['n_test']} from unseen merchants. "
                   "Real-world accuracy will differ — treat it as a demo of the method, not a guarantee.")
    st.write("**Try it**")
    a, b, c = st.columns(3)
    kind = a.selectbox("Payment type", ["buygoods", "paybill"])
    name = b.text_input("Merchant name", "GREEN VALLEY CLINIC")
    amt = c.number_input("Amount (KSh)", min_value=1.0, value=1500.0)
    label, conf, how = cat.predict_one(kind, name, "", amt)
    st.success(f"**{label}** — confidence {conf:.0%} (via {how})")
