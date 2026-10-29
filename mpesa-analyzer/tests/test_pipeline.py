import pytest

from mpesa_analyzer import analytics, generator, pipeline
from mpesa_analyzer.categorizer import Categorizer, rule_category
from mpesa_analyzer.classifier import ensure_model, train
from mpesa_analyzer.parser import parse_text


@pytest.fixture(scope="module")
def categorizer():
    return Categorizer(train())


def test_generator_roundtrip_is_lossless():
    gen = generator.generate(months=3, seed=1)
    parsed, unparsed = parse_text(generator.sms_text(gen))
    assert unparsed == [] and len(parsed) == len(gen)
    m = gen.merge(parsed, on="code", suffixes=("_g", "_p"))
    assert len(m) == len(gen)
    assert (m["amount_g"] == m["amount_p"]).all()
    assert (m["type_g"] == m["type_p"]).all()
    assert (m["balance_g"] - m["balance_p"]).abs().max() < 0.01
    assert (m["fee_g"] == m["fee_p"]).all()
    assert (m["party_g"] == m["party_p"]).all()


def test_balances_never_negative():
    assert (generator.generate(months=6, seed=3)["balance"] >= 0).all()


def test_rules():
    assert rule_category("paybill", "KPLC PREPAID", "123") == "Utilities"
    assert rule_category("buygoods", "NAIVAS SUPERMARKET") == "Groceries"
    assert rule_category("paybill", "HOMELAND PROPERTIES", "A12 RENT") == "Rent"
    assert rule_category("sent", "JOHN KAMAU") == "Person-to-Person"
    assert rule_category("buygoods", "MOUNTAIN VIEW CLINIC") is None


def test_classifier_handles_rule_misses(categorizer):
    cat, conf, method = categorizer.predict_one("buygoods", "MOUNTAIN VIEW CLINIC", "", 1800)
    assert method == "ml" and cat == "Health"


def test_end_to_end_accuracy(categorizer):
    gen = generator.generate(months=6, seed=11)
    df, _ = pipeline.from_text(generator.sms_text(gen), categorizer)
    truth = gen.set_index("code")["true_category"]
    acc = (df["category"].values == truth.loc[df["code"]].values).mean()
    assert acc > 0.9
    assert {"rule", "ml"} <= set(df["method"])


def test_analytics_functions_run(categorizer):
    df, _ = pipeline.from_synthetic(3, 5, categorizer)
    k = analytics.kpis(df)
    assert k["inflow"] > 0 and k["spend"] > 0 and k["count"] == len(df)
    assert not analytics.spend_by_category(df).empty
    assert not analytics.monthly_category(df).empty
    assert not analytics.daily_flow(df).empty
    assert analytics.weekday_hour(df).shape == (7, 24)
    assert not analytics.top_counterparties(df).empty
    assert not analytics.fees_by_month(df).empty
    assert not analytics.balance_series(df).empty
    assert "party" in analytics.recurring_payments(df).columns


def test_statement_pipeline(categorizer):
    from mpesa_analyzer.config import DATA_DIR
    df, _ = pipeline.from_statement(DATA_DIR / "sample_statement.csv", categorizer)
    assert not df.empty and "category" in df
    assert analytics.kpis(df)["fees"] == 13.0


def test_ensure_model_trains_when_missing(tmp_path):
    bundle = ensure_model(path=tmp_path / "m.joblib")
    assert (tmp_path / "m.joblib").exists() and "pipeline" in bundle
