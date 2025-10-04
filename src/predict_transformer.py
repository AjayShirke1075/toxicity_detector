# src/predict_transformer.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(ROOT, "models", "transformer_model")

def load_model_and_tokenizer(model_dir):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        print(f"Loaded transformer from local folder: {model_dir}")
        return tokenizer, model
    except Exception as e:
        print(f"Could not load local transformer ({model_dir}): {e}")
        print("Falling back to 'distilbert-base-uncased' from Hugging Face hub (not fine-tuned).")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
        return tokenizer, model

tokenizer, model = load_model_and_tokenizer(MODEL_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def predict_text_transformer(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    score = probs[0][1].item()
    label = "toxic" if score >= 0.5 else "not_toxic"
    return {"label": label, "score": score, "text": text}
