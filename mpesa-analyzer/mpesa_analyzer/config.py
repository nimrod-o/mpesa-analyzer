from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "classifier.joblib"

CATEGORIES = [
    "Income", "Cash Deposit", "Cash Withdrawal", "Airtime & Data", "Person-to-Person",
    "Rent", "Utilities", "Groceries", "Transport", "Food & Dining", "Entertainment",
    "Betting", "Health", "Education", "Savings & Investments", "Fees & Charges", "Other",
]

# Canonical column order of every parsed transaction table.
COLUMNS = [
    "code", "timestamp", "type", "direction", "amount",
    "party", "account", "phone", "balance", "fee", "raw",
]
