# app/streamlit_app.py
import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import os
import io
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from datetime import datetime

# explanation helpers from src
from src.preprocess import clean_text
from src.explain import token_contributions_from_pipeline, make_shap_linear_explainer, shap_values_for_text

# ---------------------------
# Dark Tech Theme CSS
# ---------------------------
DARK_TECH_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1a2e 50%, #16213e 100%);
        font-family: 'Rajdhani', sans-serif;
    }
    
    /* Animated Background Grid */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            linear-gradient(rgba(0, 255, 255, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 255, 255, 0.03) 1px, transparent 1px);
        background-size: 50px 50px;
        animation: gridMove 20s linear infinite;
        pointer-events: none;
        z-index: 0;
    }
    
    @keyframes gridMove {
        0% { transform: translate(0, 0); }
        100% { transform: translate(50px, 50px); }
    }
    
    /* Main Title */
    h1 {
        font-family: 'Orbitron', sans-serif !important;
        background: linear-gradient(135deg, #00f5ff 0%, #00d4ff 50%, #0099ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 30px rgba(0, 245, 255, 0.5);
        font-weight: 900 !important;
        letter-spacing: 3px;
        animation: titleGlow 3s ease-in-out infinite;
        margin-bottom: 2rem !important;
    }
    
    @keyframes titleGlow {
        0%, 100% { filter: drop-shadow(0 0 10px rgba(0, 245, 255, 0.5)); }
        50% { filter: drop-shadow(0 0 20px rgba(0, 245, 255, 0.8)); }
    }
    
    /* Subheaders */
    h2, h3 {
        font-family: 'Orbitron', sans-serif !important;
        color: #00f5ff !important;
        text-shadow: 0 0 10px rgba(0, 245, 255, 0.3);
        font-weight: 700 !important;
        letter-spacing: 2px;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(10, 14, 39, 0.95) 0%, rgba(22, 33, 62, 0.95) 100%);
        backdrop-filter: blur(10px);
        border-right: 2px solid rgba(0, 245, 255, 0.2);
        box-shadow: 5px 0 20px rgba(0, 245, 255, 0.1);
    }
    
    [data-testid="stSidebar"] h2 {
        color: #00f5ff !important;
        border-bottom: 2px solid rgba(0, 245, 255, 0.3);
        padding-bottom: 10px;
    }
    
    /* Cards/Containers */
    .element-container, [data-testid="stVerticalBlock"] > div {
        background: rgba(26, 26, 46, 0.6);
        border-radius: 15px;
        padding: 15px;
        border: 1px solid rgba(0, 245, 255, 0.2);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .element-container:hover {
        border-color: rgba(0, 245, 255, 0.5);
        box-shadow: 0 0 20px rgba(0, 245, 255, 0.2);
        transform: translateY(-2px);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00f5ff 0%, #0099ff 100%);
        color: #0a0e27;
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 12px 30px;
        font-size: 16px;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(0, 245, 255, 0.3);
        text-transform: uppercase;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0, 245, 255, 0.5);
        background: linear-gradient(135deg, #00d4ff 0%, #0088ee 100%);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
    }
    
    /* Text Input & Text Area */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: rgba(10, 14, 39, 0.8) !important;
        border: 2px solid rgba(0, 245, 255, 0.3) !important;
        border-radius: 10px !important;
        color: #00f5ff !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-size: 16px !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: rgba(0, 245, 255, 0.8) !important;
        box-shadow: 0 0 15px rgba(0, 245, 255, 0.3) !important;
    }
    
    /* Selectbox */
    .stSelectbox > div > div > div {
        background: rgba(10, 14, 39, 0.8) !important;
        border: 2px solid rgba(0, 245, 255, 0.3) !important;
        border-radius: 10px !important;
        color: #00f5ff !important;
    }
    
    /* Slider */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #00f5ff 0%, #0099ff 100%) !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-family: 'Orbitron', sans-serif !important;
        font-size: 2.5rem !important;
        font-weight: 900 !important;
        background: linear-gradient(135deg, #00f5ff 0%, #ff00ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: metricPulse 2s ease-in-out infinite;
    }
    
    @keyframes metricPulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    
    [data-testid="stMetricLabel"] {
        color: #00f5ff !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
    }
    
    [data-testid="stMetricDelta"] {
        font-family: 'Orbitron', sans-serif !important;
        font-weight: 700 !important;
    }
    
    /* DataFrames */
    [data-testid="stDataFrame"] {
        background: rgba(10, 14, 39, 0.8) !important;
        border: 1px solid rgba(0, 245, 255, 0.3) !important;
        border-radius: 10px !important;
    }
    
    /* Charts */
    .vega-embed {
        background: rgba(10, 14, 39, 0.6) !important;
        border-radius: 15px !important;
        padding: 15px !important;
        border: 1px solid rgba(0, 245, 255, 0.2) !important;
    }
    
    /* Info/Warning/Error boxes */
    .stAlert {
        background: rgba(26, 26, 46, 0.8) !important;
        border-left: 4px solid #00f5ff !important;
        border-radius: 10px !important;
        backdrop-filter: blur(10px) !important;
    }
    
    /* Download Button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #ff00ff 0%, #ff0099 100%);
        color: white;
        font-family: 'Orbitron', sans-serif;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(255, 0, 255, 0.3);
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(255, 0, 255, 0.5);
    }
    
    /* Checkbox */
    .stCheckbox {
        color: #00f5ff !important;
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(10, 14, 39, 0.6) !important;
        border: 2px dashed rgba(0, 245, 255, 0.3) !important;
        border-radius: 15px !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(0, 245, 255, 0.6) !important;
        background: rgba(10, 14, 39, 0.8) !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(10, 14, 39, 0.5);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #00f5ff 0%, #0099ff 100%);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #00d4ff 0%, #0088ee 100%);
    }
    
    /* Token Highlight Enhancement */
    .token-highlight {
        display: inline-block;
        padding: 4px 8px;
        margin: 2px;
        border-radius: 6px;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 500;
        transition: all 0.2s ease;
        animation: tokenFadeIn 0.5s ease;
    }
    
    @keyframes tokenFadeIn {
        from { opacity: 0; transform: scale(0.9); }
        to { opacity: 1; transform: scale(1); }
    }
    
    .token-highlight:hover {
        transform: scale(1.1);
        box-shadow: 0 0 15px rgba(0, 245, 255, 0.5);
    }
    
    /* Stats Card */
    .stats-card {
        background: linear-gradient(135deg, rgba(0, 245, 255, 0.1) 0%, rgba(0, 153, 255, 0.1) 100%);
        border: 2px solid rgba(0, 245, 255, 0.3);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .stats-card:hover {
        border-color: rgba(0, 245, 255, 0.6);
        box-shadow: 0 0 30px rgba(0, 245, 255, 0.2);
        transform: translateY(-5px);
    }
    
    /* Pulse Animation for Important Elements */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    .pulse {
        animation: pulse 2s ease-in-out infinite;
    }
    
    /* Text Colors */
    p, label, span {
        color: #b8c5d6 !important;
    }
    
    strong {
        color: #00f5ff !important;
    }
</style>
"""

# ---------------------------
# Enhanced Token Highlighting
# ---------------------------
def highlight_text_html_enhanced(pipe, text):
    """Enhanced token highlighting with dark tech theme"""
    toks = get_token_importance_tf(pipe, text)
    if not toks:
        return f"<div style='color: #b8c5d6; white-space:pre-wrap; font-family: Rajdhani;'>{clean_text(text)}</div>"
    
    scores = np.array([t[-1] for t in toks])
    max_abs = max(1e-9, float(np.max(np.abs(scores))))
    spans = []
    
    for token, coef, tfidf_val, contrib in toks:
        intensity = abs(contrib) / max_abs
        if contrib >= 0:
            # Toxic - red/pink gradient
            color = f"rgba(255, 0, 100, {0.2 + 0.6*intensity})"
            border = f"1px solid rgba(255, 0, 100, {0.5 + 0.5*intensity})"
            glow = f"0 0 10px rgba(255, 0, 100, {0.3*intensity})"
        else:
            # Safe - cyan/green gradient
            color = f"rgba(0, 245, 255, {0.2 + 0.6*intensity})"
            border = f"1px solid rgba(0, 245, 255, {0.5 + 0.5*intensity})"
            glow = f"0 0 10px rgba(0, 245, 255, {0.3*intensity})"
        
        spans.append(
            f"<span class='token-highlight' style='background:{color}; border:{border}; "
            f"box-shadow:{glow}; font-weight: 600;'>{token}</span>"
        )
    
    return "<div style='line-height: 2.2; padding: 15px; background: rgba(10, 14, 39, 0.6); border-radius: 10px;'>" + " ".join(spans) + "</div>"

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
st.set_page_config(
    page_title="🛡️ Toxicity Detector — AI-Powered Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Dark Tech Theme
st.markdown(DARK_TECH_CSS, unsafe_allow_html=True)

# Header with animated title
st.markdown("""
    <h1 style='text-align:center; font-size: 3.5rem; margin-top: 1rem;'>
        🛡️ TOXICITY DETECTOR
    </h1>
    <p style='text-align:center; color: #00f5ff; font-size: 1.2rem; font-family: Rajdhani; letter-spacing: 2px; margin-bottom: 2rem;'>
        AI-POWERED CONTENT ANALYSIS & MODERATION SYSTEM
    </p>
""", unsafe_allow_html=True)

# Initialize session state
if "history" not in st.session_state:
    st.session_state["history"] = []
if "total_analyzed" not in st.session_state:
    st.session_state["total_analyzed"] = 0
if "toxic_count" not in st.session_state:
    st.session_state["toxic_count"] = 0

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ CONFIGURATION")
    st.markdown("---")
    
    model_choice = st.selectbox(
        "🤖 Model Selection",
        ["Baseline (TF-IDF)", "Transformer (DistilBERT)"],
        help="Choose the AI model for toxicity detection"
    )
    
    threshold = st.slider(
        "🎯 Decision Threshold",
        0.0, 1.0, 0.5, 0.01,
        help="Adjust sensitivity of toxicity detection"
    )
    
    st.markdown("---")
    st.markdown("### 📝 QUICK EXAMPLES")
    
    example = st.selectbox("Select a test case:", [
        "I will kill you",
        "You are an idiot",
        "Thanks for your help!",
        "I love your work",
        "Please go away and die"
    ])
    
    st.markdown("---")
    st.markdown("### 📊 STATISTICS")
    
    # Stats display
    st.markdown(f"""
        <div class='stats-card'>
            <p style='margin: 0; font-size: 0.9rem;'>Total Analyzed</p>
            <p style='margin: 0; font-size: 1.8rem; font-weight: 700; color: #00f5ff;'>{st.session_state['total_analyzed']}</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
        <div class='stats-card'>
            <p style='margin: 0; font-size: 0.9rem;'>Toxic Detected</p>
            <p style='margin: 0; font-size: 1.8rem; font-weight: 700; color: #ff0099;'>{st.session_state['toxic_count']}</p>
        </div>
    """, unsafe_allow_html=True)
    
    if st.session_state['total_analyzed'] > 0:
        toxic_rate = (st.session_state['toxic_count'] / st.session_state['total_analyzed']) * 100
        st.markdown(f"""
            <div class='stats-card'>
                <p style='margin: 0; font-size: 0.9rem;'>Toxicity Rate</p>
                <p style='margin: 0; font-size: 1.8rem; font-weight: 700; color: #ffaa00;'>{toxic_rate:.1f}%</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📁 BATCH PROCESSING")
    uploaded = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="CSV must contain 'comment_text' column"
    )

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
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.markdown("### 💬 TEXT ANALYSIS")
    
    text = st.text_area(
        "Enter text to analyze for toxicity:",
        value=example,
        height=180,
        placeholder="Type or paste your text here..."
    )
    
    analyze = st.button("🔍 ANALYZE TEXT", use_container_width=True)

    if analyze:
        if not text or text.strip() == "":
            st.warning("⚠️ Please enter some text to analyze.")
        else:
            with st.spinner("🔄 Analyzing content..."):
                if model_choice == "Baseline (TF-IDF)":
                    if baseline_pipe is None:
                        st.error("❌ Baseline model not found. Please train the model or select Transformer.")
                    else:
                        res = predict_baseline(baseline_pipe, text)
                        
                        # Update statistics
                        st.session_state["total_analyzed"] += 1
                        if res["label"] == "Toxic":
                            st.session_state["toxic_count"] += 1
                        
                        # Display result with enhanced styling
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            st.metric(
                                label="🎯 Prediction",
                                value=res["label"],
                                delta="High Risk" if res["score"] > 0.7 else "Low Risk" if res["score"] < 0.3 else "Medium Risk"
                            )
                        
                        with col_b:
                            st.metric(
                                label="📊 Confidence Score",
                                value=f"{res['score']:.1%}",
                                delta=f"{res['score']:.3f}"
                            )
                        
                        with col_c:
                            st.metric(
                                label="🤖 Model",
                                value="TF-IDF",
                                delta="Baseline"
                            )

                        st.markdown("---")
                        
                        # Show explanations
                        if st.checkbox("🔬 Show Detailed Token Analysis", value=True):
                            try:
                                df_expl = token_contributions_from_pipeline(baseline_pipe, text, top_n=30)
                                if df_expl.empty:
                                    st.info("ℹ️ No significant tokens found for analysis.")
                                else:
                                    st.markdown("#### 📈 Token Contribution Chart")
                                    
                                    # Enhanced chart with dark theme
                                    chart = alt.Chart(df_expl.reset_index()).mark_bar().encode(
                                        x=alt.X('contrib:Q', title='Contribution Score'),
                                        y=alt.Y('token:N', sort='-x', title='Token'),
                                        color=alt.condition(
                                            alt.datum.contrib > 0,
                                            alt.value('#ff0066'),  # Toxic - pink/red
                                            alt.value('#00f5ff')   # Safe - cyan
                                        ),
                                        tooltip=['token', 'contrib', 'tfidf', 'coef']
                                    ).properties(
                                        width=600,
                                        height=400
                                    ).configure_axis(
                                        labelColor='#00f5ff',
                                        titleColor='#00f5ff',
                                        gridColor='rgba(0, 245, 255, 0.1)'
                                    ).configure_view(
                                        strokeWidth=0
                                    )
                                    
                                    st.altair_chart(chart, use_container_width=True)

                                    st.markdown("#### 🎨 Highlighted Token Visualization")
                                    st.markdown("**Color Legend:** <span style='color: #ff0066;'>■ Toxic Indicators</span> | <span style='color: #00f5ff;'>■ Safe Indicators</span>", unsafe_allow_html=True)
                                    
                                    try:
                                        html = highlight_text_html_enhanced(baseline_pipe, text)
                                        st.markdown(html, unsafe_allow_html=True)
                                    except Exception:
                                        st.dataframe(df_expl, use_container_width=True)

                                    csv_bytes = df_expl[['token','tfidf','coef','contrib']].to_csv(index=False).encode('utf-8')
                                    st.download_button(
                                        "📥 Download Token Analysis (CSV)",
                                        data=csv_bytes,
                                        file_name=f"token_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv"
                                    )

                                if st.checkbox("🧪 Compute SHAP Analysis (Advanced)", value=False):
                                    with st.spinner("Computing SHAP values..."):
                                        try:
                                            expl = make_shap_linear_explainer(baseline_pipe)
                                            shap_df = shap_values_for_text(expl, baseline_pipe, text)
                                            if shap_df.empty:
                                                st.info("ℹ️ SHAP analysis returned no results.")
                                            else:
                                                st.markdown("#### 🔬 SHAP Values")
                                                st.dataframe(shap_df.head(40), use_container_width=True)
                                        except Exception as e:
                                            st.error(f"❌ SHAP analysis failed: {e}")
                            except Exception as e:
                                st.error(f"❌ Analysis error: {e}")

                        # Add to history
                        hist = st.session_state.get("history", [])
                        hist.insert(0, {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "text": text[:100] + "..." if len(text) > 100 else text,
                            "label": res["label"],
                            "score": res["score"],
                            "model": "Baseline"
                        })
                        st.session_state["history"] = hist[:50]

                else:
                    if transformer is None:
                        st.error("❌ Transformer model not available.")
                    else:
                        import torch as _torch
                        res = predict_transformer_local(transformer, text, device)
                        
                        # Update statistics
                        st.session_state["total_analyzed"] += 1
                        if res["label"] == "Toxic":
                            st.session_state["toxic_count"] += 1
                        
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            st.metric(
                                label="🎯 Prediction",
                                value=res["label"],
                                delta="High Risk" if res["score"] > 0.7 else "Low Risk" if res["score"] < 0.3 else "Medium Risk"
                            )
                        
                        with col_b:
                            st.metric(
                                label="📊 Confidence Score",
                                value=f"{res['score']:.1%}",
                                delta=f"{res['score']:.3f}"
                            )
                        
                        with col_c:
                            st.metric(
                                label="🤖 Model",
                                value="DistilBERT",
                                delta="Transformer"
                            )
                        
                        st.info("ℹ️ Token-level explanations for transformers require SHAP/Captum integration.")
                        
                        hist = st.session_state.get("history", [])
                        hist.insert(0, {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "text": text[:100] + "..." if len(text) > 100 else text,
                            "label": res["label"],
                            "score": res["score"],
                            "model": "Transformer"
                        })
                        st.session_state["history"] = hist[:50]

    st.markdown("---")
    st.markdown("### 📜 ANALYSIS HISTORY")
    
    history_df = pd.DataFrame(st.session_state["history"])
    if not history_df.empty:
        st.dataframe(
            history_df,
            use_container_width=True,
            height=300
        )
        
        csv_bytes = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download History (CSV)",
            data=csv_bytes,
            file_name=f"analysis_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("📭 No analysis history yet. Start analyzing text to build your history.")

with col2:
    st.markdown("### 📊 BATCH PROCESSING")
    
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            if "comment_text" not in df.columns:
                st.error("❌ CSV must contain a 'comment_text' column.")
            else:
                st.success(f"✅ Loaded {len(df)} rows successfully!")
                
                with st.spinner("🔄 Processing batch..."):
                    if model_choice == "Baseline (TF-IDF)":
                        if baseline_pipe is None:
                            st.error("❌ Baseline model not found.")
                        else:
                            df["cleaned"] = df["comment_text"].astype(str).apply(clean_text)
                            df["score"] = df["cleaned"].apply(lambda t: float(baseline_pipe.predict_proba([t])[0][1]))
                            df["label"] = df["score"].apply(lambda s: "Toxic" if s >= threshold else "Not Toxic")
                            
                            st.markdown("#### 📋 Results Preview")
                            st.dataframe(df[["comment_text","label","score"]].head(50), use_container_width=True)
                    else:
                        if transformer is None:
                            st.error("❌ Transformer model not available.")
                        else:
                            import torch as _torch
                            device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
                            scores = []
                            progress_bar = st.progress(0)
                            for idx, t in enumerate(df["comment_text"].astype(str).tolist()):
                                r = predict_transformer_local(transformer, t, device)
                                scores.append(r["score"])
                                progress_bar.progress((idx + 1) / len(df))
                            df["score"] = scores
                            df["label"] = df["score"].apply(lambda s: "Toxic" if s >= threshold else "Not Toxic")
                            
                            st.markdown("#### 📋 Results Preview")
                            st.dataframe(df[["comment_text","label","score"]].head(50), use_container_width=True)

                    csv_bytes = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Download Batch Results",
                        data=csv_bytes,
                        file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        except Exception as e:
            st.error(f"❌ Failed to process file: {e}")
    else:
        st.info("📤 Upload a CSV file to begin batch processing")

    st.markdown("---")
    st.markdown("### 📈 SCORE DISTRIBUTION")
    
    hist_df = None
    if not history_df.empty:
        hist_df = history_df
    elif 'df' in locals():
        hist_df = df
        
    if hist_df is not None and "score" in hist_df.columns:
        chart = alt.Chart(pd.DataFrame(hist_df["score"])).mark_bar().encode(
            x=alt.X('score:Q', bin=alt.Bin(maxbins=20), title='Toxicity Score'),
            y=alt.Y('count()', title='Frequency'),
            color=alt.value('#00f5ff'),
            tooltip=['count()']
        ).properties(
            width=300,
            height=250
        ).configure_axis(
            labelColor='#00f5ff',
            titleColor='#00f5ff',
            gridColor='rgba(0, 245, 255, 0.1)'
        ).configure_view(
            strokeWidth=0
        )
        
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("📊 No data available for visualization yet.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 20px; font-family: Rajdhani;'>
        <p style='color: #00f5ff; font-size: 1.1rem; margin-bottom: 10px;'>
            ⚡ Powered by Advanced Machine Learning & Natural Language Processing
        </p>
        <p style='color: #b8c5d6; font-size: 0.9rem;'>
            Built with ❤️ using TF-IDF Baseline & DistilBERT Transformer Models
        </p>
        <p style='color: #888; font-size: 0.8rem; margin-top: 10px;'>
            For enhanced transformer explanations, consider integrating SHAP or Captum libraries
        </p>
    </div>
""", unsafe_allow_html=True)

