# Toxicity Detector

A Python-based NLP project to detect toxic comments in text using **TF-IDF + Logistic Regression** and **Transformer-based models**. The project provides both a **Tkinter GUI** and a **Streamlit web app**, allowing real-time text analysis and interpretability through **SHAP explanations**.

---

## Problem Statement

Online platforms are often flooded with toxic comments, hate speech, or abusive language. Detecting such content automatically is crucial for **maintaining safe and healthy online communities**. This project addresses the need for **accurate and interpretable toxicity detection**.

---

## Features

- **Baseline Model:** TF-IDF vectorizer + Logistic Regression classifier  
- **Advanced Model:** Transformer-based sequence classification (DistilBERT)  
- **Interactive UI:** Tkinter desktop app & Streamlit web interface  
- **Explainability:** Token-level contributions using SHAP and TF-IDF weights  
- **CSV Export:** Download token contributions for analysis  
- **Threshold Customization:** Adjustable toxicity threshold for predictions  

---

## Dataset

- **Civil Comments** dataset from Google / Hugging Face  
- Contains labeled comments with toxicity scores  
- Used to train and evaluate the models  

---

## Installation

```bash
# Clone the repository
git clone https://github.com/AjayShirke1075/toxicity_detector.git
cd toxicity_detector

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate   # Windows
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app/streamlit_app.py

# Or run Tkinter GUI
python app/tk_ui.py
