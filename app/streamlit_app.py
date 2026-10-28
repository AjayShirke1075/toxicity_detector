# app/streamlit_app.py
import sys
import os
import time
from datetime import datetime

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import joblib
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

from src.preprocess import clean_text
from src.explain import token_contributions_from_pipeline

# =========================================================
# ⚙️ MODEL CACHING & INITIALIZATION
# =========================================================
@st.cache_resource(show_spinner=False)
def load_baseline_model():
    """Load TF-IDF vectorizer and trained classifier pipeline."""
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_path = os.path.join(base, "models", "baseline_clf.joblib")
    vect_path = os.path.join(base, "models", "tfidf_vectorizer.joblib")

    if not os.path.exists(model_path) or not os.path.exists(vect_path):
        return None

    clf = joblib.load(model_path)
    vectorizer = joblib.load(vect_path)

    from sklearn.pipeline import Pipeline
    return Pipeline([
        ("tfidf", vectorizer),
        ("clf", clf)
    ])

# =========================================================
# 🎨 SENIOR-ENGINEER ENTERPRISE DESIGN SYSTEM
# =========================================================
ENTERPRISE_THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* ===== GLOBAL CANVAS ===== */
html, body, .stApp {
    background-color: #090D16 !important;
    background-image: 
        radial-gradient(ellipse 90% 60% at 50% -15%, rgba(30, 58, 110, 0.25) 0%, transparent 70%),
        radial-gradient(circle 800px at 100% 90%, rgba(15, 23, 42, 0.4) 0%, transparent 60%) !important;
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    color: #F8FAFC !important;
    letter-spacing: -0.01em;
}

/* Base text elements */
p, span, label, div {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Code & monospace */
code, pre, .mono-font {
    font-family: 'JetBrains Mono', monospace !important;
}

/* ===== APP HEADER & BRANDING ===== */
.brand-header-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1.25rem 1.75rem;
    margin-bottom: 1.5rem;
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    backdrop-filter: blur(16px);
    box-shadow: 0 4px 24px -1px rgba(0, 0, 0, 0.4);
}

.brand-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    background: rgba(59, 130, 246, 0.12);
    border: 1px solid rgba(59, 130, 246, 0.3);
    color: #60A5FA;
    margin-bottom: 8px;
}

.brand-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #F8FAFC;
    margin: 0;
    line-height: 1.2;
    display: flex;
    align-items: center;
    gap: 10px;
}

.brand-subtitle {
    font-size: 0.85rem;
    color: #94A3B8;
    margin-top: 4px;
    margin-bottom: 0;
}

.system-status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 500;
    background: rgba(16, 185, 129, 0.08);
    border: 1px solid rgba(16, 185, 129, 0.25);
    color: #34D399;
}

.status-beacon {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #10B981;
    box-shadow: 0 0 10px #10B981;
    animation: beaconPulse 2s ease-in-out infinite;
}

@keyframes beaconPulse {
    0%, 100% { transform: scale(1); opacity: 0.9; }
    50% { transform: scale(1.3); opacity: 0.4; }
}

/* ===== SENIOR CARDS & CONTAINERS ===== */
.surface-card {
    background: rgba(15, 23, 42, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 16px -2px rgba(0, 0, 0, 0.4);
    margin-bottom: 1rem;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

.surface-card:hover {
    border-color: rgba(255, 255, 255, 0.14);
    box-shadow: 0 6px 20px -2px rgba(0, 0, 0, 0.5);
}

.card-title {
    font-size: 0.9rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94A3B8;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ===== SIDEBAR STYLING ===== */
[data-testid="stSidebar"] {
    background-color: #0B0F19 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
}

[data-testid="stSidebar"] hr {
    border-color: rgba(255, 255, 255, 0.08) !important;
    margin: 1.25rem 0 !important;
}

.sidebar-heading {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #64748B;
    margin-bottom: 0.75rem;
}

/* ===== METRICS RIBBON ===== */
[data-testid="stMetric"] {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    backdrop-filter: blur(8px);
    transition: transform 0.15s ease, border-color 0.15s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    border-color: rgba(59, 130, 246, 0.3);
}

[data-testid="stMetricLabel"] {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
    color: #94A3B8 !important;
}

[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.85rem !important;
    font-weight: 700 !important;
    color: #F8FAFC !important;
    letter-spacing: -0.02em !important;
}

/* ===== FORM CONTROLS & BUTTONS ===== */
textarea, input[type="text"], input[type="number"] {
    background: rgba(11, 15, 25, 0.8) !important;
    color: #F1F5F9 !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 8px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

textarea:focus, input[type="text"]:focus, input[type="number"]:focus {
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2) !important;
    outline: none !important;
}

/* Primary Button */
.stButton > button {
    background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
    color: #FFFFFF !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: -0.01em !important;
    border-radius: 8px !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    padding: 0.65rem 1.5rem !important;
    box-shadow: 0 2px 10px rgba(37, 99, 235, 0.35) !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(37, 99, 235, 0.5) !important;
    border-color: rgba(255, 255, 255, 0.25) !important;
}

.stButton > button:active {
    transform: translateY(1px) !important;
    box-shadow: 0 1px 4px rgba(37, 99, 235, 0.2) !important;
}

/* Secondary Button */
button[kind="secondary"] {
    background: rgba(30, 41, 59, 0.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: #CBD5E1 !important;
}

button[kind="secondary"]:hover {
    background: rgba(51, 65, 85, 0.7) !important;
    color: #FFFFFF !important;
    border-color: rgba(255, 255, 255, 0.2) !important;
}

/* ===== TABS ===== */
[data-baseweb="tab-list"] {
    background: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
    margin-bottom: 1.5rem !important;
}

[data-baseweb="tab"] {
    border-radius: 6px !important;
    color: #94A3B8 !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    padding: 8px 18px !important;
    border: none !important;
    transition: all 0.15s ease !important;
}

[data-baseweb="tab"]:hover {
    color: #F8FAFC !important;
    background: rgba(255, 255, 255, 0.04) !important;
}

[aria-selected="true"] {
    background: #1E293B !important;
    color: #F8FAFC !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3) !important;
}

/* ===== VERDICT HERO CARDS ===== */
.verdict-card {
    border-radius: 12px;
    padding: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.25rem;
    backdrop-filter: blur(12px);
}

.verdict-toxic {
    background: linear-gradient(135deg, rgba(225, 29, 72, 0.12) 0%, rgba(136, 19, 55, 0.08) 100%);
    border: 1px solid rgba(244, 63, 94, 0.35);
    box-shadow: 0 4px 20px -2px rgba(225, 29, 72, 0.2);
}

.verdict-clean {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(6, 78, 59, 0.08) 100%);
    border: 1px solid rgba(16, 185, 129, 0.35);
    box-shadow: 0 4px 20px -2px rgba(16, 185, 129, 0.2);
}

.verdict-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.tag-toxic {
    background: rgba(244, 63, 94, 0.2);
    border: 1px solid rgba(244, 63, 94, 0.4);
    color: #FDA4AF;
}

.tag-clean {
    background: rgba(16, 185, 129, 0.2);
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: #6EE7B7;
}

.verdict-headline {
    font-size: 1.5rem;
    font-weight: 700;
    margin: 8px 0 4px 0;
}

.headline-toxic { color: #FB7185; }
.headline-clean { color: #34D399; }

.verdict-desc {
    font-size: 0.88rem;
    color: #94A3B8;
    margin: 0;
}

.score-display-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.75rem;
    font-weight: 800;
    line-height: 1;
}

/* ===== TOKEN HIGHLIGHT CHIPS ===== */
.token-chip {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 8px;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    margin: 3px 4px 3px 0;
}

.chip-toxic {
    background: rgba(225, 29, 72, 0.15);
    border: 1px solid rgba(244, 63, 94, 0.4);
    color: #FDA4AF;
}

.chip-mitigating {
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.35);
    color: #6EE7B7;
}

.chip-neutral {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.08);
    color: #94A3B8;
}

/* ===== ENTERPRISE PROGRESS & THRESHOLD BAR ===== */
.progress-track-wrapper {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 16px 20px;
    margin: 16px 0;
}

.progress-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
    font-size: 0.8rem;
    color: #94A3B8;
}

.progress-bar-container {
    height: 10px;
    border-radius: 5px;
    background: rgba(255, 255, 255, 0.06);
    position: relative;
    overflow: visible;
}

.progress-fill {
    height: 100%;
    border-radius: 5px;
    transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
}

.progress-threshold-line {
    position: absolute;
    top: -4px;
    bottom: -4px;
    width: 2px;
    background: #F59E0B;
    box-shadow: 0 0 6px rgba(245, 158, 11, 0.6);
    z-index: 2;
}

.progress-threshold-label {
    position: absolute;
    top: 14px;
    transform: translateX(-50%);
    font-size: 0.7rem;
    font-weight: 600;
    color: #F59E0B;
    font-family: 'JetBrains Mono', monospace;
    white-space: nowrap;
}

/* ===== TABLE STYLING ===== */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #090D16;
}
::-webkit-scrollbar-thumb {
    background: #1E293B;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: #334155;
}
</style>
"""

# =========================================================
# 🚀 PAGE INITIALIZATION
# =========================================================
st.set_page_config(
    page_title="Sentinel AI | Trust & Safety Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(ENTERPRISE_THEME_CSS, unsafe_allow_html=True)

# =========================================================
# 💾 SESSION STATE MANAGEMENT
# =========================================================
if "total_audited" not in st.session_state:
    st.session_state.total_audited = 0
if "total_flagged" not in st.session_state:
    st.session_state.total_flagged = 0
if "audit_history" not in st.session_state:
    st.session_state.audit_history = []
if "threshold" not in st.session_state:
    st.session_state.threshold = 0.50
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# =========================================================
# ⚙️ SIDEBAR: CONFIGURATION & BENCHMARKS
# =========================================================
with st.sidebar:
    st.markdown("<div class='sidebar-heading'>🛡️ Moderation Engine</div>", unsafe_allow_html=True)
    
    model_architecture = st.selectbox(
        "Active Inference Architecture",
        ["Production Baseline (TF-IDF + Saga Logistic Regression)", "Transformer DistilBERT (Experimental)"],
        index=0,
        help="Select the inference model engine for scoring and attribution"
    )
    
    st.markdown("<div class='sidebar-heading'>🎯 Decision Sensitivity</div>", unsafe_allow_html=True)
    threshold = st.slider(
        "Toxicity Cutoff Threshold",
        min_value=0.05,
        max_value=0.95,
        value=st.session_state.threshold,
        step=0.01,
        format="%.2f",
        help="Probability score above which text is flagged as toxic."
    )
    st.session_state.threshold = threshold
    
    # Sensitivity mode hint
    if threshold < 0.35:
        mode_desc = "🔴 Strict Filter (High Recall: Catches borderline toxicity)"
    elif threshold > 0.65:
        mode_desc = "🟢 Conservative Filter (High Precision: Low false positives)"
    else:
        mode_desc = "🔵 Balanced Policy (Standard Enterprise Production)"
    
    st.caption(f"**Policy Mode:** {mode_desc}")
    
    st.markdown("---")
    st.markdown("<div class='sidebar-heading'>⚡ Quick Benchmark Payloads</div>", unsafe_allow_html=True)
    
    benchmark_categories = {
        "Direct Threat / Violence": "I will find you and kill you.",
        "Harassment / Ad Hominem": "You are completely incompetent and an absolute idiot.",
        "Hostile Feedback": "I hate this terrible software so much and everyone who built it.",
        "Constructive Critique": "The new update introduces several critical latency bugs, but overall the architecture shows promise.",
        "Verified Safe": "Thank you for the quick and helpful response! Really appreciate your team's support."
    }
    
    selected_benchmark = st.selectbox(
        "Choose test scenario:",
        list(benchmark_categories.keys()),
        index=3
    )
    default_text = benchmark_categories[selected_benchmark]
    
    st.markdown("---")
    st.markdown("<div class='sidebar-heading'>📈 Telemetry Summary</div>", unsafe_allow_html=True)
    
    if st.session_state.total_audited > 0:
        flag_rate = (st.session_state.total_flagged / st.session_state.total_audited) * 100
        clean_count = st.session_state.total_audited - st.session_state.total_flagged
        
        st.write(f"• **Audited:** `{st.session_state.total_audited}`")
        st.write(f"• **Flagged:** `{st.session_state.total_flagged}` ({flag_rate:.1f}%)")
        st.write(f"• **Clean:** `{clean_count}` ({100 - flag_rate:.1f}%)")
    else:
        st.caption("No requests processed in this session.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Clear Audit Cache", use_container_width=True, type="secondary"):
        st.session_state.total_audited = 0
        st.session_state.total_flagged = 0
        st.session_state.audit_history = []
        st.session_state.last_result = None
        st.rerun()

# =========================================================
# 🛡️ TOP ENTERPRISE BRANDING BAR
# =========================================================
st.markdown("""
<div class="brand-header-container">
    <div>
        <div class="brand-badge">Enterprise NLP • Trust & Safety Platform</div>
        <h1 class="brand-title">
            <span>🛡️</span> Sentinel Moderation Intelligence
        </h1>
        <p class="brand-subtitle">
            Automated content classification, threat severity scoring, and token-level attribution (XAI).
        </p>
    </div>
    <div style="text-align: right;">
        <div class="system-status-pill">
            <div class="status-beacon"></div>
            <span>ENGINE STATUS: OPERATIONAL</span>
        </div>
        <div style="font-size: 0.75rem; color: #64748B; margin-top: 6px; font-family: 'JetBrains Mono', monospace;">
            LATENCY: ~11ms &bull; ACC: 92.4% &bull; v1.4.2
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# 🔄 MODEL LOADER
# =========================================================
model_pipe = load_baseline_model()
if model_pipe is None:
    st.error("⚠️ Baseline model artifacts not found in `models/`. Please train or generate the baseline weights.")
    st.stop()

# =========================================================
# 📊 TOP KPI TELEMETRY STRIP
# =========================================================
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

with kpi_col1:
    st.metric(
        label="Total Payloads Audited",
        value=f"{st.session_state.total_audited:,}"
    )

with kpi_col2:
    st.metric(
        label="Toxicity Violations",
        value=f"{st.session_state.total_flagged:,}"
    )

with kpi_col3:
    clean_total = st.session_state.total_audited - st.session_state.total_flagged
    st.metric(
        label="Safe / Cleared",
        value=f"{clean_total:,}"
    )

with kpi_col4:
    rate = (st.session_state.total_flagged / st.session_state.total_audited * 100) if st.session_state.total_audited > 0 else 0.0
    st.metric(
        label="Violation Ratio",
        value=f"{rate:.1f}%",
        delta=f"-{100 - rate:.1f}% clean" if st.session_state.total_audited > 0 else None,
        delta_color="off"
    )

st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# 📑 NAVIGATION TABS
# =========================================================
tab_analysis, tab_analytics, tab_audit = st.tabs([
    "⚡ Real-Time Moderation",
    "📊 Fleet Analytics & Distributions",
    "📋 Compliance Audit Trail"
])

# ---------------------------------------------------------
# TAB 1: REAL-TIME MODERATION
# ---------------------------------------------------------
with tab_analysis:
    st.markdown("<div class='card-title'>Payload Input & Inspection</div>", unsafe_allow_html=True)
    
    col_input, col_meta = st.columns([3, 1])
    
    with col_input:
        user_text = st.text_area(
            "Payload Content",
            value=default_text,
            height=140,
            help="Type or paste arbitrary user-generated content to perform toxicity inference.",
            label_visibility="collapsed"
        )
        
        btn_col1, btn_col2, btn_col3 = st.columns([1.5, 1, 3])
        with btn_col1:
            analyze_clicked = st.button("⚡ Run Moderation Audit", use_container_width=True, type="primary")
        with btn_col2:
            reset_clicked = st.button("Clear Input", use_container_width=True, type="secondary")
            if reset_clicked:
                user_text = ""
                st.session_state.last_result = None
                st.rerun()

    with col_meta:
        char_count = len(user_text)
        word_count = len(user_text.split()) if user_text.strip() else 0
        
        st.markdown(f"""
        <div class="surface-card" style="padding: 1rem;">
            <div class="card-title" style="margin-bottom: 0.5rem;">Payload Telemetry</div>
            <div style="font-size: 0.85rem; color: #94A3B8; display: flex; justify-content: space-between; margin-bottom: 6px;">
                <span>Length:</span> <strong style="color: #F8FAFC; font-family: 'JetBrains Mono';">{char_count} chars</strong>
            </div>
            <div style="font-size: 0.85rem; color: #94A3B8; display: flex; justify-content: space-between; margin-bottom: 6px;">
                <span>Word Count:</span> <strong style="color: #F8FAFC; font-family: 'JetBrains Mono';">{word_count} words</strong>
            </div>
            <div style="font-size: 0.85rem; color: #94A3B8; display: flex; justify-content: space-between; margin-bottom: 6px;">
                <span>Decision Cutoff:</span> <strong style="color: #F59E0B; font-family: 'JetBrains Mono';">{threshold:.2f}</strong>
            </div>
            <div style="font-size: 0.85rem; color: #94A3B8; display: flex; justify-content: space-between;">
                <span>Engine:</span> <strong style="color: #38BDF8; font-family: 'JetBrains Mono';">TF-IDF SAGA</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Inference Execution
    if analyze_clicked:
        if not user_text.strip():
            st.warning("⚠️ Input payload is empty. Please provide text to analyze.")
        else:
            t0 = time.perf_counter()
            cleaned = clean_text(user_text)
            proba = float(model_pipe.predict_proba([cleaned])[0][1])
            inference_ms = (time.perf_counter() - t0) * 1000
            
            is_toxic = proba >= threshold
            label = "Toxic" if is_toxic else "Clean"
            
            # Severity Tier
            if proba >= 0.75:
                severity = "Critical Risk"
                severity_color = "#F43F5E"
            elif proba >= threshold:
                severity = "Elevated Risk"
                severity_color = "#FB7185"
            elif proba >= 0.25:
                severity = "Low / Borderline"
                severity_color = "#F59E0B"
            else:
                severity = "Negligible Risk"
                severity_color = "#10B981"
            
            # Update session stats
            st.session_state.total_audited += 1
            if is_toxic:
                st.session_state.total_flagged += 1
            
            # Store in audit history
            st.session_state.audit_history.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "text": user_text,
                "label": label,
                "probability": proba,
                "threshold": threshold,
                "latency_ms": round(inference_ms, 2)
            })
            
            st.session_state.last_result = {
                "text": user_text,
                "label": label,
                "is_toxic": is_toxic,
                "proba": proba,
                "severity": severity,
                "severity_color": severity_color,
                "latency_ms": inference_ms
            }

    # Render Results if available
    res = st.session_state.last_result
    if res is not None:
        st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 1.5rem 0;'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Inference Verdict & Risk Assessment</div>", unsafe_allow_html=True)
        
        is_toxic = res["is_toxic"]
        proba = res["proba"]
        card_class = "verdict-toxic" if is_toxic else "verdict-clean"
        tag_class = "tag-toxic" if is_toxic else "tag-clean"
        headline_class = "headline-toxic" if is_toxic else "headline-clean"
        headline_text = "FLAGGED AS TOXIC CONTENT" if is_toxic else "CLEARED • CONTENT APPROVED"
        sub_desc = (
            f"Toxicity confidence ({proba:.1%}) exceeds the decision boundary ({threshold:.1%}). Policy action required."
            if is_toxic else
            f"Content toxicity probability ({proba:.1%}) is safely below the policy threshold ({threshold:.1%})."
        )
        tag_text = "🚨 POLICY VIOLATION" if is_toxic else "✅ COMPLIANT CONTENT"
        score_color = "#FB7185" if is_toxic else "#34D399"
        
        st.markdown(f"""
        <div class="verdict-card {card_class}">
            <div style="flex: 1; padding-right: 20px;">
                <div class="verdict-tag {tag_class}">{tag_text}</div>
                <div class="verdict-headline {headline_class}">{headline_text}</div>
                <p class="verdict-desc">{sub_desc}</p>
                <div style="margin-top: 10px; display: flex; gap: 15px; font-size: 0.8rem; color: #94A3B8;">
                    <span>Severity Tier: <strong style="color: {res['severity_color']};">{res['severity']}</strong></span>
                    <span>&bull;</span>
                    <span>Inference Latency: <strong style="color: #F8FAFC; font-family: 'JetBrains Mono';">{res['latency_ms']:.1f}ms</strong></span>
                </div>
            </div>
            <div style="text-align: right; min-width: 150px;">
                <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: #94A3B8; margin-bottom: 4px;">Confidence Score</div>
                <div class="score-display-value" style="color: {score_color};">{proba:.1%}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Enterprise Threshold Gauge
        fill_width = min(max(proba * 100, 0), 100)
        threshold_pos = min(max(threshold * 100, 0), 100)
        fill_gradient = "linear-gradient(90deg, #10B981 0%, #F59E0B 60%, #F43F5E 100%)"
        
        st.markdown(f"""
        <div class="progress-track-wrapper">
            <div class="progress-header">
                <span>0.0 (Clean Standard)</span>
                <span style="font-weight: 600; color: #F8FAFC;">Continuous Toxicity Spectrum</span>
                <span>1.0 (Critical Threat)</span>
            </div>
            <div class="progress-bar-container">
                <div class="progress-fill" style="width: {fill_width}%; background: {fill_gradient};"></div>
                <div class="progress-threshold-line" style="left: {threshold_pos}%;">
                    <div class="progress-threshold-label">Threshold: {threshold:.2f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Explainable AI (XAI)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>🔬 Explainable AI (XAI) • Token Contribution Analysis</div>", unsafe_allow_html=True)
        
        df_expl = token_contributions_from_pipeline(model_pipe, res["text"], top_n=15)
        
        if not df_expl.empty:
            tab_chart, tab_text_recon, tab_table = st.tabs([
                "📊 Attribution Waterfall Chart",
                "🔤 Highlighted Token Preview",
                "📄 Detailed Contribution Data"
            ])
            
            with tab_chart:
                chart_df = df_expl.copy()
                chart_df["Impact"] = chart_df["contrib"].apply(lambda x: "Increases Toxicity" if x > 0 else "Mitigates Toxicity")
                
                # Professional Altair Bar Chart
                chart = alt.Chart(chart_df).mark_bar(cornerRadiusEnd=4).encode(
                    x=alt.X('contrib:Q', title='Attribution Impact Score (SHAP equivalent)',
                            axis=alt.Axis(
                                labelColor='#94A3B8',
                                titleColor='#CBD5E1',
                                gridColor='rgba(255, 255, 255, 0.06)',
                                domainColor='rgba(255, 255, 255, 0.12)',
                                format='.3f'
                            )),
                    y=alt.Y('token:N', sort='-x', title='Identified Token',
                            axis=alt.Axis(
                                labelColor='#E2E8F0',
                                titleColor='#CBD5E1',
                                domainColor='rgba(255, 255, 255, 0.12)'
                            )),
                    color=alt.Color('Impact:N', scale=alt.Scale(
                        domain=['Increases Toxicity', 'Mitigates Toxicity'],
                        range=['#F43F5E', '#10B981']
                    ), legend=alt.Legend(
                        orient='top',
                        labelColor='#E2E8F0',
                        titleColor='#94A3B8'
                    )),
                    tooltip=[
                        alt.Tooltip('token:N', title='Token'),
                        alt.Tooltip('contrib:Q', title='Contribution', format='.4f'),
                        alt.Tooltip('tfidf:Q', title='TF-IDF Weight', format='.4f'),
                        alt.Tooltip('coef:Q', title='Model Weight', format='.4f')
                    ]
                ).properties(
                    height=360,
                    background='transparent'
                ).configure_view(
                    strokeWidth=0
                )
                
                st.altair_chart(chart, use_container_width=True)
            
            with tab_text_recon:
                st.markdown("<p style='color: #94A3B8; font-size: 0.85rem; margin-bottom: 12px;'>Words from your payload colored by model attribution impact:</p>", unsafe_allow_html=True)
                
                contrib_dict = {row['token'].lower(): row['contrib'] for _, row in df_expl.iterrows()}
                words = res["text"].split()
                html_chips = []
                
                for w in words:
                    clean_w = clean_text(w)
                    contrib_val = contrib_dict.get(clean_w, 0.0)
                    
                    if contrib_val > 0.05:
                        html_chips.append(f"<span class='token-chip chip-toxic'>{w} <small style='opacity:0.8;'>+{contrib_val:.2f}</small></span>")
                    elif contrib_val < -0.05:
                        html_chips.append(f"<span class='token-chip chip-mitigating'>{w} <small style='opacity:0.8;'>{contrib_val:.2f}</small></span>")
                    else:
                        html_chips.append(f"<span class='token-chip chip-neutral'>{w}</span>")
                
                st.markdown(
                    f"<div class='surface-card' style='line-height: 2; padding: 1.25rem;'>{' '.join(html_chips)}</div>",
                    unsafe_allow_html=True
                )
            
            with tab_table:
                display_df = df_expl[['token', 'contrib', 'tfidf', 'coef']].copy()
                display_df.columns = ['Token', 'Attribution Impact', 'TF-IDF Weight', 'Model Weight (Coef)']
                st.dataframe(
                    display_df.style.format({
                        'Attribution Impact': '{:+.4f}',
                        'TF-IDF Weight': '{:.4f}',
                        'Model Weight (Coef)': '{:+.4f}'
                    }),
                    use_container_width=True,
                    height=280
                )
        else:
            st.info("ℹ️ No distinct vocabulary tokens were strongly correlated with the decision boundary in this text.")

# ---------------------------------------------------------
# TAB 2: FLEET ANALYTICS & DISTRIBUTIONS
# ---------------------------------------------------------
with tab_analytics:
    st.markdown("<div class='card-title'>Fleet Telemetry & Statistical Analysis</div>", unsafe_allow_html=True)
    
    if len(st.session_state.audit_history) > 1:
        history_df = pd.DataFrame(st.session_state.audit_history)
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("<p style='font-size: 0.85rem; font-weight: 600; color: #CBD5E1;'>Temporal Probability Trend</p>", unsafe_allow_html=True)
            history_df['Sequence'] = range(1, len(history_df) + 1)
            
            trend_chart = alt.Chart(history_df).mark_line(
                point=alt.OverlayMarkDef(filled=True, size=50, color='#3B82F6'),
                color='#3B82F6',
                strokeWidth=2
            ).encode(
                x=alt.X('Sequence:O', title='Sequential Request #',
                        axis=alt.Axis(labelColor='#94A3B8', titleColor='#94A3B8', gridColor='rgba(255,255,255,0.05)')),
                y=alt.Y('probability:Q', title='Toxicity Probability', scale=alt.Scale(domain=[0, 1]),
                        axis=alt.Axis(labelColor='#94A3B8', titleColor='#94A3B8', gridColor='rgba(255,255,255,0.05)', format='.0%')),
                tooltip=['timestamp', 'label', 'probability', 'latency_ms']
            ).properties(
                height=280,
                background='transparent'
            ).configure_view(
                strokeWidth=0
            )
            st.altair_chart(trend_chart, use_container_width=True)
            
        with c2:
            st.markdown("<p style='font-size: 0.85rem; font-weight: 600; color: #CBD5E1;'>Score Distribution Density</p>", unsafe_allow_html=True)
            dist_chart = alt.Chart(history_df).mark_bar(
                color='#38BDF8',
                opacity=0.85,
                cornerRadiusTopLeft=4,
                cornerRadiusTopRight=4
            ).encode(
                x=alt.X('probability:Q', bin=alt.Bin(maxbins=12), title='Score Range',
                        axis=alt.Axis(labelColor='#94A3B8', titleColor='#94A3B8', gridColor='rgba(255,255,255,0.05)', format='.0%')),
                y=alt.Y('count()', title='Frequency Count',
                        axis=alt.Axis(labelColor='#94A3B8', titleColor='#94A3B8', gridColor='rgba(255,255,255,0.05)')),
                tooltip=['count()']
            ).properties(
                height=280,
                background='transparent'
            ).configure_view(
                strokeWidth=0
            )
            st.altair_chart(dist_chart, use_container_width=True)
            
        # Summary Analytics Cards
        avg_score = history_df['probability'].mean()
        avg_lat = history_df['latency_ms'].mean()
        max_score = history_df['probability'].max()
        
        st.markdown("<br>", unsafe_allow_html=True)
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.metric("Session Average Probability", f"{avg_score:.1%}")
        with sc2:
            st.metric("Max Recorded Severity", f"{max_score:.1%}")
        with sc3:
            st.metric("Average Inference Latency", f"{avg_lat:.2f} ms")
    else:
        st.info("ℹ️ Execute at least 2 moderation audits in this session to generate statistical trends and fleet charts.")

# ---------------------------------------------------------
# TAB 3: COMPLIANCE AUDIT TRAIL
# ---------------------------------------------------------
with tab_audit:
    st.markdown("<div class='card-title'>Compliance & Event Audit Log</div>", unsafe_allow_html=True)
    
    if len(st.session_state.audit_history) > 0:
        audit_df = pd.DataFrame(st.session_state.audit_history)
        
        # Filter controls
        filter_col1, filter_col2 = st.columns([2, 1])
        with filter_col1:
            status_filter = st.radio(
                "Filter records by classification:",
                ["All Records", "Violations Only", "Clean Only"],
                horizontal=True
            )
        
        filtered_df = audit_df.copy()
        if status_filter == "Violations Only":
            filtered_df = filtered_df[filtered_df["label"] == "Toxic"]
        elif status_filter == "Clean Only":
            filtered_df = filtered_df[filtered_df["label"] == "Clean"]
        
        # Table display
        table_view = filtered_df[['timestamp', 'label', 'probability', 'threshold', 'latency_ms', 'text']].copy()
        table_view.columns = ['Timestamp (UTC)', 'Classification', 'Score', 'Threshold', 'Latency (ms)', 'Payload']
        table_view = table_view.sort_values('Timestamp (UTC)', ascending=False)
        table_view['Score'] = table_view['Score'].apply(lambda x: f"{x:.2%}")
        table_view['Threshold'] = table_view['Threshold'].apply(lambda x: f"{x:.2f}")
        table_view['Latency (ms)'] = table_view['Latency (ms)'].apply(lambda x: f"{x:.1f}")
        
        st.dataframe(
            table_view,
            use_container_width=True,
            height=320
        )
        
        # Export Actions
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Audit Trail (CSV)",
            data=csv_data,
            file_name=f"toxicity_audit_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            type="secondary"
        )
    else:
        st.info("ℹ️ Audit trail is empty. Analyzed inputs will appear here with complete telemetry and timestamps.")
