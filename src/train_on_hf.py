import argparse
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
import os

def train_and_save_model(dataset_name="google/civil_comments", sample_frac=0.05, output_path="models/toxic_model.pkl"):
    # 1. Load dataset
    print(f"Loading dataset '{dataset_name}'...")
    dataset = load_dataset(dataset_name, split="train")

    # Convert to pandas
    df = pd.DataFrame(dataset)
    df = df.sample(frac=sample_frac, random_state=42)

    X = df["text"]  # CivilComments column
    y = (df["toxicity"] >= 0.5).astype(int)  # 1 = toxic, 0 = not toxic

    # 2. Pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
        ("clf", LogisticRegression(max_iter=200)),
    ])

    # 3. Train
    print("Training model...")
    pipeline.fit(X, y)

    # 4. Evaluate
    preds = pipeline.predict(X)
    print(classification_report(y, preds))

    # 5. Save model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(pipeline, output_path)
    print(f"✅ Model saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="google/civil_comments")
    parser.add_argument("--sample_frac", type=float, default=0.05)
    args = parser.parse_args()

    train_and_save_model(dataset_name=args.dataset, sample_frac=args.sample_frac)
