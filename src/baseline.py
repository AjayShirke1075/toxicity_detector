# src/baseline.py
"""
Train a TF-IDF + LogisticRegression baseline.
If --data is provided (CSV) it will use that file. Otherwise uses a small toy dataset.
Saves two artifacts into models/: tfidf_vectorizer.joblib and baseline_clf.joblib
Run as: python -m src.baseline  OR  python -m src.baseline --data data/raw/train.csv
"""
import argparse
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def get_toy_data():
    tox = [
        "i will kill you",
        "you are a idiot",
        "shut up and die",
        "i will punch you",
        "i hate you",
        "go to hell",
    ]
    nontox = [
        "hello, how are you?",
        "thank you for your help",
        "looking forward to meeting you",
        "that was a nice movie",
        "have a good day",
        "see you later",
    ]
    texts = tox + nontox
    labels = [1]*len(tox) + [0]*len(nontox)
    return pd.DataFrame({"comment_text": texts, "toxic": labels})

def load_csv(path):
    df = pd.read_csv(path)
    if 'comment_text' not in df.columns or 'toxic' not in df.columns:
        raise ValueError("CSV must contain 'comment_text' and 'toxic' columns")
    df = df[['comment_text','toxic']].dropna(subset=['comment_text'])
    return df

def train_and_save(df, vect_out, model_out):
    X = df['comment_text'].astype(str).values
    y = df['toxic'].astype(int).values
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    vect = TfidfVectorizer(max_features=20000, ngram_range=(1,2), stop_words='english')
    Xtr = vect.fit_transform(X_train)
    Xval = vect.transform(X_val)

    clf = LogisticRegression(max_iter=2000, class_weight='balanced', solver='saga')
    clf.fit(Xtr, y_train)

    preds = clf.predict(Xval)
    probs = clf.predict_proba(Xval)[:,1]

    print("Validation classification report:")
    print(classification_report(y_val, preds))
    try:
        print("ROC AUC:", roc_auc_score(y_val, probs))
    except Exception:
        pass

    joblib.dump(vect, vect_out)
    joblib.dump(clf, model_out)
    print(f"Saved vectorizer -> {vect_out}")
    print(f"Saved classifier -> {model_out}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None, help="Path to CSV with columns 'comment_text' and 'toxic'")
    parser.add_argument("--vector_out", type=str, default=os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    parser.add_argument("--model_out", type=str, default=os.path.join(MODELS_DIR, "baseline_clf.joblib"))
    args = parser.parse_args()

    if args.data:
        print("Loading dataset from:", args.data)
        df = load_csv(args.data)
    else:
        print("No dataset provided — using small toy dataset for quick demo.")
        df = get_toy_data()

    train_and_save(df, args.vector_out, args.model_out)

if __name__ == "__main__":
    main()
