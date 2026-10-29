from datetime import datetime

from mpesa_analyzer.parser import parse_sms, parse_statement, parse_text
from mpesa_analyzer.config import DATA_DIR

SAMPLES = {
    "received": "QGH7A1B2C3 Confirmed. You have received Ksh1,500.00 from JOHN KAMAU 0712345678 on 5/3/26 at 2:15 PM. New M-PESA balance is Ksh3,200.50. Transaction cost, Ksh0.00.",
    "sent": "QGH7A1B2C4 Confirmed. Ksh500.00 sent to JANE WANJIKU 0722345678 on 5/3/26 at 3:00 PM. New M-PESA balance is Ksh2,700.50. Transaction cost, Ksh7.00.",
    "paybill": "QGH7A1B2C5 Confirmed. Ksh2,000.00 sent to KPLC PREPAID for account 12345678 on 6/3/26 at 8:10 AM. New M-PESA balance is Ksh700.50. Transaction cost, Ksh0.00.",
    "buygoods": "QGH7A1B2C6 Confirmed. Ksh350.00 paid to NAIVAS SUPERMARKET. on 6/3/26 at 6:20 PM.New M-PESA balance is Ksh350.50. Transaction cost, Ksh0.00.",
    "airtime": "QGH7A1B2C7 Confirmed.You bought Ksh50.00 of airtime on 7/3/26 at 9:00 AM.New M-PESA balance is Ksh300.50. Transaction cost, Ksh0.00.",
    "withdraw": "QGH7A1B2C8 Confirmed. on 7/3/26 at 10:00 AM Withdraw Ksh200.00 from 123456 - MAMA JOY AGENT NAIROBI New M-PESA balance is Ksh100.50. Transaction cost, Ksh11.00.",
    "deposit": "QGH7A1B2C9 Confirmed. On 7/3/26 at 11:30 AM Give Ksh1,000.00 cash to TOTO FINANCIAL SERVICES New M-PESA balance is Ksh1,100.50. Transaction cost, Ksh0.00.",
}


def test_each_message_type_is_recognised():
    for kind, sms in SAMPLES.items():
        rec = parse_sms(sms)
        assert rec is not None, kind
        assert rec["type"] == kind


def test_received_fields():
    rec = parse_sms(SAMPLES["received"])
    assert rec["amount"] == 1500.0 and rec["party"] == "JOHN KAMAU"
    assert rec["balance"] == 3200.50 and rec["direction"] == "in"
    assert rec["timestamp"] == datetime(2026, 3, 5, 14, 15)


def test_paybill_account_and_fee():
    rec = parse_sms(SAMPLES["paybill"])
    assert rec["account"] == "12345678" and rec["party"] == "KPLC PREPAID"
    assert parse_sms(SAMPLES["sent"])["fee"] == 7.0


def test_text_block_splits_and_reports_unparsed():
    blob = "Random promo text: win big!\n\n" + "\n\n".join(SAMPLES.values())
    df, unparsed = parse_text(blob)
    assert len(df) == 7
    assert len(unparsed) == 1 and "promo" in unparsed[0]


def test_garbage_and_empty_input():
    df, unparsed = parse_text("hello world")
    assert df.empty and len(unparsed) == 1
    df, unparsed = parse_text("")
    assert df.empty and unparsed == []


def test_statement_csv():
    df = parse_statement(DATA_DIR / "sample_statement.csv")
    assert len(df) == 9  # the one Failed row is excluded
    assert set(df["type"]) >= {"received", "paybill", "buygoods", "sent", "fee", "airtime", "withdraw"}
    assert df.loc[df["type"] == "paybill", "account"].iloc[0] == "A12 RENT"
