# app/streamlit_app.py
import os
import io
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

# explanation helpers from src
from src.preprocess import clean_text
from src.explain import token_contributions_from_pipeline, make_shap_linear_explainer, shap_values_for_text

# ---------------------------
# Small helpers (highlight HTML)
# ---------------------------
def get_token_importance_tf(pipe, text):
    """Return list[(token, score)] using pipeline's tfidf and linear coef (simple fallback)."""
    try:
        vect = pipe.named_steps.get("tfidf") or pipe.named_steps.get("vectorizer")
        clf = pipe.named_steps.get("clf") or pipe.named_steps.get("classifier") or pipe.named_steps.get("logisticregression")
    except Exception:
        # try scanning
        vect = None
        clf = None
        for step in getattr(pipe, "named_steps", {}).values():
            if hasattr(step, "vocabulary_") and vect is None:
                vect = step
            if hasattr(step, "coef_") and clf is None:
                clf = step
    if vect is None or clf is None:
        return []

    cleaned = clean_text(text)
    X = vect.transform([cleaned]).toarray()[0]
    inv_vocab = {idx: tok for tok, idx in vect.vocabulary_.items()}
    coefs = clf.coef_[0]
    toks = []
    for idx, val in enumerate(X):
        if val == 0:
            continue
        token = inv_vocab.get(idx)
        if token:
            toks.append((token, float(coefs[idx]), float(val), float(coefs[idx]*val)))
    return toks

def highlight_text_html_tf(pipe, text):
    toks = get_token_importance_tf(pipe, text)
    if not toks:
        return "<div style='white-space:pre-wrap'>{}</div>".format(clean_text(text))
    scores = np.array([t[-1] for t in toks])
    max_abs = max(1e-9, float(np.max(np.abs(scores))))
    spans = []
    for token, coef, tfidf_val, contrib in toks:
        intensity = abs(contrib) / max_abs
        if contrib >= 0:
            color = f"rgba(255,0,0,{0.15 + 0.6*intensity})"
        else:
            color = f"rgba(0,200,0,{0.15 + 0.6*intensity})"
        spans.append(f"<span style='background:{color};padding:2px;border-radius:4px;margin-right:2px'>{token}</span>")
    return "<div style='line-height:1.7;'>" + " ".join(spans) + "</div>"

# ---------------------------
# Model loaders (cached)
# ---------------------------
@st.cache_resource
def load_baseline_model(path="models/toxic_model.pkl"):
    if os.path.exists(path):
        try:
            pipe = joblib.load(path)
            return pipe
        except Exception as e:
            st.warning(f"Could not load baseline model from {path}: {e}")
    return None

@st.cache_resource
def load_transformer_model(local_dir="models/transformer_model"):
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        if os.path.isdir(local_dir) and os.listdir(local_dir):
            tokenizer = AutoTokenizer.from_pretrained(local_dir)
            model = AutoModelForSequenceClassification.from_pretrained(local_dir)
            return {"tokenizer": tokenizer, "model": model, "local": True}
        else:
            tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
            return {"tokenizer": tokenizer, "model": model, "local": False}
    except Exception:
        return None

# ---------------------------
# Prediction helpers
# ---------------------------
def predict_baseline(pipe, text):
    cleaned = clean_text(text)
    proba = pipe.predict_proba([cleaned])[0][1]
    label = "Toxic" if proba >= 0.5 else "Not Toxic"
    return {"label": label, "score": float(proba), "cleaned": cleaned}

def predict_transformer_local(transformer, text, device):
    tokenizer = transformer["tokenizer"]
    model = transformer["model"]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    model.to(device)
    model.eval()
    with __import__("torch").no_grad():
        outputs = model(**inputs)
        probs = __import__("torch").nn.functional.softmax(outputs.logits, dim=-1)
    score = float(probs[0][1].item())
    label = "Toxic" if score >= 0.5 else "Not Toxic"
    return {"label": label, "score": score, "cleaned": clean_text(text)}

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Toxicity Detector — Enhanced", layout="wide")
st.markdown("<h1 style='text-align:center'>Toxicity Detector (Enhanced)</h1>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Settings & Models")
    model_choice = st.selectbox("Model", ["Baseline (TF-IDF)", "Transformer (DistilBERT)"])
    threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)
    dark = st.checkbox("Dark mode (experimental)", value=False)
    st.write("---")
    st.markdown("**Quick examples**")
    example = st.selectbox("Choose example", [
        "I will kill you",
        "You are an idiot",
        "Thanks for your help!",
        "I love your work",
        "Please go away and die"
    ])
    st.write("---")
    st.markdown("**Batch**")
    uploaded = st.file_uploader("Upload CSV (one column 'comment_text')", type=["csv"])
    st.write("Download predictions after upload.")

# Load models
baseline_pipe = load_baseline_model()
transformer = None
device = None
if model_choice.startswith("Transformer"):
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        transformer = load_transformer_model()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        transformer = None

# Main layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Input")
    text = st.text_area("Enter comment to analyze", value=example, height=180)
    st.write("Use the Analyze button or upload a CSV for batch.")
    analyze = st.button("Analyze")

    if analyze:
        if not text or text.strip() == "":
            st.warning("Please enter some text.")
        else:
            if model_choice == "Baseline (TF-IDF)":
                if baseline_pipe is None:
                    st.error("Baseline model not found. Train baseline or select Transformer.")
                else:
                    res = predict_baseline(baseline_pipe, text)
                    st.metric(label=f"Prediction ({model_choice})", value=res["label"], delta=f"{res['score']:.3f}")

                    # -------------------------
                    # Show explanations (TF-IDF contributions)
                    # -------------------------
                    if st.checkbox("Show token-level explanation (TF-IDF / SHAP-like)", value=True):
                        try:
                            df_expl = token_contributions_from_pipeline(baseline_pipe, text, top_n=30)
                            if df_expl.empty:
                                st.info("No tokens with non-zero TF-IDF found for this text.")
                            else:
                                chart = alt.Chart(df_expl.reset_index()).mark_bar().encode(
                                    x=alt.X('contrib:Q', title='Contribution (weight * tfidf)'),
                                    y=alt.Y('token:N', sort='-x', title='Token')
                                ).properties(width=500, height=300)
                                st.altair_chart(chart)

                                st.markdown("**Highlighted tokens (positive = red, negative = green)**")
                                try:
                                    html = highlight_text_html_tf(baseline_pipe, text)
                                    st.markdown(html, unsafe_allow_html=True)
                                except Exception:
                                    st.write(df_expl)

                                csv_bytes = df_expl[['token','tfidf','coef','contrib']].to_csv(index=False).encode('utf-8')
                                st.download_button("Download token contributions (CSV)", data=csv_bytes, file_name="token_contributions.csv", mime="text/csv")

                            if st.checkbox("Also compute true SHAP LinearExplainer (slower)", value=False):
                                try:
                                    expl = make_shap_linear_explainer(baseline_pipe)
                                    shap_df = shap_values_for_text(expl, baseline_pipe, text)
                                    if shap_df.empty:
                                        st.info("SHAP explanation returned empty result.")
                                    else:
                                        st.dataframe(shap_df.head(40))
                                except Exception as e:
                                    st.error(f"SHAP explainer failed: {e}")
                        except Exception as e:
                            st.error(f"Explanation error: {e}")

                    # add to history
                    hist = st.session_state.get("history", [])
                    hist.insert(0, {"text": text, "label": res["label"], "score": res["score"], "model": "baseline"})
                    st.session_state["history"] = hist[:50]

            else:
                if transformer is None:
                    st.error("Transformer model not available.")
                else:
                    import torch as _torch
                    res = predict_transformer_local(transformer, text, device)
                    st.metric(label=f"Prediction ({'Transformer — local' if transformer.get('local') else 'Transformer — hub'})", value=res["label"], delta=f"{res['score']:.3f}")
                    st.write("Note: token-level explanations for transformers require SHAP/Captum (not included here).")
                    hist = st.session_state.get("history", [])
                    hist.insert(0, {"text": text, "label": res["label"], "score": res["score"], "model": "transformer"})
                    st.session_state["history"] = hist[:50]

    st.write("---")
    st.subheader("Prediction history (most recent)")
    if "history" not in st.session_state:
        st.session_state["history"] = []
    history_df = pd.DataFrame(st.session_state["history"])
    if not history_df.empty:
        st.dataframe(history_df)
    else:
        st.write("No predictions yet. Try analyzing some text.")

with col2:
    st.subheader("Batch / CSV")
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            if "comment_text" not in df.columns:
                st.error("CSV must contain a 'comment_text' column.")
            else:
                st.write(f"Loaded {len(df)} rows.")
                if model_choice == "Baseline (TF-IDF)":
                    if baseline_pipe is None:
                        st.error("Baseline model not found — choose Transformer or train baseline.")
                    else:
                        df["cleaned"] = df["comment_text"].astype(str).apply(clean_text)
                        df["score"] = df["cleaned"].apply(lambda t: float(baseline_pipe.predict_proba([t])[0][1]))
                        df["label"] = df["score"].apply(lambda s: "Toxic" if s >= threshold else "Not Toxic")
                        st.dataframe(df[["comment_text","label","score"]].head(50))
                else:
                    if transformer is None:
                        st.error("Transformer model not available.")
                    else:
                        import torch as _torch
                        device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
                        scores = []
                        for t in df["comment_text"].astype(str).tolist():
                            r = predict_transformer_local(transformer, t, device)
                            scores.append(r["score"])
                        df["score"] = scores
                        df["label"] = df["score"].apply(lambda s: "Toxic" if s >= threshold else "Not Toxic")
                        st.dataframe(df[["comment_text","label","score"]].head(50))

                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")

    st.write("---")
    st.subheader("Probability distribution")
    hist_df = None
    if not history_df.empty:
        hist_df = history_df
    elif 'df' in locals():
        hist_df = df
    if hist_df is not None and "score" in hist_df.columns:
        chart = alt.Chart(pd.DataFrame(hist_df["score"])).mark_bar().encode(
            x=alt.X('score:Q', bin=alt.Bin(maxbins=20), title='Toxicity score'),
            y='count()'
        ).properties(width=300, height=200)
        st.altair_chart(chart)
    else:
        st.write("No score data to display yet.")

st.markdown("---")
st.markdown("Built with ❤️ — baseline TF-IDF + optional Transformer. For token-level explanations with transformer, consider SHAP/Captum integration.")
