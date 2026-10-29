"""Fallback ML classifier for merchants that the regex rules do not recognise."""
from __future__ import annotations

import math
import re
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline

from .config import MODEL_DIR, MODEL_PATH
from .generator import generate

TRAINABLE_TYPES = ("paybill", "buygoods")


def row_text(kind: str, party: str, account: str, amount: float) -> str:
    bucket = int(math.log10(max(float(amount), 1.0))) if amount == amount else 0
    account = re.sub(r"\d+", " ", str(account or ""))
    return f"{kind} {party} {account} amt{bucket}".lower()


def _make_pipeline():
    return make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True),
        LogisticRegression(max_iter=2000, C=8.0),
    )


def training_frame(months: int = 24, seed: int = 7) -> pd.DataFrame:
    df = generate(months=months, seed=seed)
    df = df[df["type"].isin(TRAINABLE_TYPES)].copy()
    df["category"] = df["true_category"]
    return df.reset_index(drop=True)


def train(df: pd.DataFrame | None = None, seed: int = 42) -> dict:
    """Train the classifier. Holdout accuracy is measured on merchants unseen in training."""
    df = training_frame() if df is None else df
    X = np.array([row_text(r.type, r.party, r.account, r.amount) for r in df.itertuples()], dtype=object)
    y = df["category"].to_numpy()
    groups = df["party"].to_numpy()
    # Hold out ~30% of merchants *within each category* so every class is still seen in training,
    # but the test merchants are names the model has never encountered.
    rng = np.random.RandomState(seed)
    test_parties: set[str] = set()
    for _, g in df.groupby("category"):
        parties = sorted(g["party"].unique())
        if len(parties) >= 3:
            k = max(1, round(len(parties) * 0.3))
            test_parties |= set(rng.choice(parties, size=k, replace=False))
    is_test = np.isin(groups, list(test_parties))
    tr, te = np.where(~is_test)[0], np.where(is_test)[0]
    probe = _make_pipeline().fit(X[tr], y[tr])
    pred = probe.predict(X[te])
    metrics = {
        "holdout_accuracy": float(accuracy_score(y[te], pred)),
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "n_merchants": int(len(set(groups))),
        "report": classification_report(y[te], pred, output_dict=True, zero_division=0),
    }
    final = _make_pipeline().fit(X, y)
    return {"pipeline": final, "metrics": metrics}


def save_model(bundle: dict, path=MODEL_PATH) -> None:
    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, path)
    except OSError:
        pass  # read-only filesystem: model simply stays in memory


def ensure_model(path=MODEL_PATH, retrain: bool = False) -> dict:
    """Load the saved model, or train (and save) one if missing/corrupt/incompatible."""
    if not retrain and path.exists():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                bundle = joblib.load(path)
            if isinstance(bundle, dict) and "pipeline" in bundle:
                return bundle
        except Exception:
            pass
    bundle = train()
    save_model(bundle, path)
    return bundle


def predict(bundle: dict, texts: list[str]) -> list[tuple[str, float]]:
    pipe = bundle["pipeline"]
    proba = pipe.predict_proba(texts)
    idx = proba.argmax(axis=1)
    return [(str(pipe.classes_[i]), float(proba[n, i])) for n, i in enumerate(idx)]
