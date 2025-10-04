# src/explain.py
import os
import pandas as pd
import numpy as np
from typing import List
from .preprocess import clean_text

def _find_vect_clf(pipeline):
    """Helper to extract vectorizer and linear classifier from an sklearn Pipeline."""
    vect = None
    clf = None
    # Common names used in our pipelines
    try:
        vect = pipeline.named_steps.get("tfidf") or pipeline.named_steps.get("vectorizer")
        clf = pipeline.named_steps.get("clf") or pipeline.named_steps.get("classifier") or pipeline.named_steps.get("logisticregression")
    except Exception:
        # try to find by attributes
        for step in pipeline.named_steps.values():
            if hasattr(step, "vocabulary_") and vect is None:
                vect = step
            if hasattr(step, "coef_") and clf is None:
                clf = step
    if vect is None or clf is None:
        raise ValueError("Could not find tfidf vectorizer and linear classifier in the pipeline.")
    return vect, clf

def token_contributions_from_pipeline(pipeline, text: str, top_n: int = 25) -> pd.DataFrame:
    """
    Compute per-token contribution = coef * tfidf_value for a single text using a pipeline
    that contains a TfidfVectorizer and a linear classifier (LogisticRegression).
    Returns a DataFrame with columns: token, tfidf, coef, contrib (sorted by abs(contrib) desc).
    This is mathematically equal to the SHAP values for a linear model.
    """
    vect, clf = _find_vect_clf(pipeline)
    cleaned = clean_text(text)
    X = vect.transform([cleaned])  # sparse
    arr = X.toarray()[0]
    # inverse vocabulary: index -> token
    inv_vocab = {idx: tok for tok, idx in vect.vocabulary_.items()}
    coefs = clf.coef_[0]  # shape (n_features,)
    rows = []
    for idx, val in enumerate(arr):
        if val == 0:
            continue
        token = inv_vocab.get(idx, None)
        if token is None:
            continue
        contrib = float(coefs[idx] * val)
        rows.append((token, float(val), float(coefs[idx]), contrib))
    if not rows:
        return pd.DataFrame(columns=["token","tfidf","coef","contrib","abs_contrib"])
    df = pd.DataFrame(rows, columns=["token","tfidf","coef","contrib"])
    df["abs_contrib"] = df["contrib"].abs()
    df = df.sort_values("abs_contrib", ascending=False).reset_index(drop=True)
    return df.head(top_n)

# Optional SHAP helper (if shap is installed)
def make_shap_linear_explainer(pipeline, background_texts: List[str] = None):
    """
    Return a shap.LinearExplainer for the linear model in the pipeline.
    background_texts: list of neutral texts to use as background (small list is fine e.g. 10)
    """
    try:
        import shap
    except Exception as e:
        raise RuntimeError("shap is not installed. Install shap to use this function.") from e

    vect, clf = _find_vect_clf(pipeline)
    if background_texts is None:
        # fallback background (neutral short texts)
        background_texts = [
            "Hello", "Thank you", "I am fine", "This is good", "That's interesting",
            "Have a nice day", "See you later", "Good morning", "Well done", "Sounds good"
        ]
    Xb = vect.transform([clean_text(t) for t in background_texts])
    # For linear models LinearExplainer is exact and fast.
    explainer = shap.LinearExplainer(clf, Xb, feature_perturbation="interventional")
    return explainer

def shap_values_for_text(explainer, pipeline, text: str):
    """
    Given a shap.LinearExplainer and pipeline, compute shap values for `text`.
    Returns (tokens, shap_values_array) where shap_values_array length = number of tokens in vocab
    but we will map only non-zero tokens to tokens.
    """
    try:
        import shap
    except Exception:
        raise RuntimeError("shap is not installed.")

    vect, clf = _find_vect_clf(pipeline)
    cleaned = clean_text(text)
    X = vect.transform([cleaned])
    # shap_values may be a list (per class) for classification; LinearExplainer returns list for multi-output
    sv = explainer.shap_values(X)
    # SV could be list or array: choose class 1 if list
    if isinstance(sv, list) and len(sv) > 1:
        vals = sv[1][0]  # class 1, first sample
    else:
        vals = sv[0] if isinstance(sv, list) else sv[0]
    # map indices to tokens
    inv_vocab = {idx: tok for tok, idx in vect.vocabulary_.items()}
    # convert to dict only for non-zero features
    nonzero = {}
    arr = X.toarray()[0]
    for idx, v in enumerate(arr):
        if v != 0:
            token = inv_vocab.get(idx, None)
            if token:
                nonzero[token] = float(vals[idx])
    # return pandas DataFrame sorted
    df = pd.DataFrame([
        {"token": t, "shap": s} for t, s in nonzero.items()
    ])
    if df.empty:
        return df
    df["abs_shap"] = df["shap"].abs()
    df = df.sort_values("abs_shap", ascending=False).reset_index(drop=True)
    return df
