# src/train_transformer.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import os

# -------------------------------
# 1. Configuration
# -------------------------------
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "models/transformer_model"
EPOCHS = 1                # increase for full training
BATCH_SIZE = 16
SAMPLE_FRAC = 0.05        # fraction of dataset for quick testing

# Create output directory if not exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# 2. Load dataset
# -------------------------------
print("Loading dataset...")
dataset_full = load_dataset("google/civil_comments", split="train")
# Take a fraction for quick test
dataset = dataset_full.shuffle(seed=42).select(range(int(len(dataset_full)*SAMPLE_FRAC)))

# -------------------------------
# 3. Preprocess labels
# -------------------------------
def preprocess_labels(example):
    # Toxicity >= 0.5 is considered toxic
    example['label'] = 1 if example['toxicity'] >= 0.5 else 0
    return example

dataset = dataset.map(preprocess_labels)

# -------------------------------
# 4. Load tokenizer and model
# -------------------------------
print(f"Loading tokenizer and model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# -------------------------------
# 5. Tokenize dataset
# -------------------------------
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Remove unnecessary columns
tokenized_dataset = tokenized_dataset.remove_columns([col for col in tokenized_dataset.column_names if col not in ["input_ids", "attention_mask", "label"]])
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# -------------------------------
# 6. Training arguments
# -------------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    save_steps=500,
    save_total_limit=2,
    logging_steps=50,
    evaluation_strategy="no",
    load_best_model_at_end=False,
    report_to="none"  # disable wandb/logs for simplicity
)

# -------------------------------
# 7. Trainer
# -------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

# -------------------------------
# 8. Train model
# -------------------------------
print("Starting training...")
trainer.train()

# -------------------------------
# 9. Save model and tokenizer
# -------------------------------
print(f"Saving model and tokenizer to {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Training complete. Model ready to use!")
