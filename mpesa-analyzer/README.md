# M-Pesa Statement Analyzer

Parses M-Pesa SMS text or CSV statement exports, categorizes every transaction
(regex rules first, a small ML classifier as fallback) and shows spending patterns
in a Streamlit dashboard. **All demo data is synthetic. No real personal data is included.**

## Tech stack
| Layer | Tool |
|---|---|
| Language | Python 3.10+ |
| Parsing | `re` (regex), `datetime` |
| Data | pandas, numpy |
| ML | scikit-learn (TF-IDF char n-grams + LogisticRegression), joblib |
| Dashboard | Streamlit, Plotly |
| Tests | pytest, Streamlit `AppTest` |

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Open the URL Streamlit prints and choose **Synthetic sample** in the sidebar.
The classifier trains itself on first launch (~2 s) if `models/classifier.joblib` is missing.

Other commands:
```bash
python -m mpesa_analyzer.cli generate --months 6   # write fake SMS to data/sample_sms.txt
python -m mpesa_analyzer.cli train                 # retrain + print holdout accuracy
pip install -r requirements-dev.txt && pytest      # run the test suite
```

## How it works
1. **Parse** (`parser.py`): 7 SMS shapes (received, sent, paybill, buy goods, airtime, withdraw, deposit)
   plus statement CSVs (`Receipt No., Completion Time, Details, Paid In, Withdrawn, Balance`).
2. **Categorize** (`categorizer.py`): transaction type and keyword regexes decide most rows.
   Anything the rules miss goes to the classifier (`classifier.py`); low-confidence guesses become `Other`.
3. **Analyze** (`analytics.py`): KPIs, category/month breakdowns, top recipients, recurring payments,
   weekday x hour heatmap, fees, balance history.
4. **Explore** (`app.py`): filters, charts, transaction table, CSV download.

## Honest notes
- The classifier is trained on synthetic merchants. Holdout accuracy is measured on merchants
  never seen in training (~70%). Real-world accuracy will differ.
- Fee tariffs in the generator are illustrative, not Safaricom's current rate card.
- Message formats vary over time. Unrecognised messages are listed in the sidebar instead of being dropped silently.
- Real statements are often PDFs. Convert to CSV first (PDF parsing is a natural next step).


