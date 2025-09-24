# src/train_transformer.py
import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np

from preprocess import clean_text   # our text cleaner

# ======================
# 1. Load dataset
# ======================
# Build an absolute path to the dataset so the script works no matter the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "reply_classification_dataset.csv"
df = pd.read_csv(CSV_PATH)

# Drop missing values & clean
df.dropna(subset=["reply", "label"], inplace=True)
df["reply"] = df["reply"].apply(clean_text)

print("Class distribution:\n", df["label"].value_counts(), "\n")

# ======================
# 2. Train-test split (safe)
# ======================
if df["label"].value_counts().min() > 1:
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )
else:
    print("⚠️ Some classes have <2 samples. Using random split without stratify.")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# ======================
# 3. Convert to HF Dataset
# ======================
label2id = {label: i for i, label in enumerate(df["label"].unique())}
id2label = {i: label for label, i in label2id.items()}

train_df["label_id"] = train_df["label"].map(label2id)
test_df["label_id"] = test_df["label"].map(label2id)

train_dataset = Dataset.from_pandas(train_df[["reply", "label_id"]])
test_dataset = Dataset.from_pandas(test_df[["reply", "label_id"]])

# ======================
# 4. Tokenizer
# ======================
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["reply"], padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# Rename to "labels" for HuggingFace Trainer
train_dataset = train_dataset.rename_column("label_id", "labels")
test_dataset = test_dataset.rename_column("label_id", "labels")

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# ======================
# 5. Model
# ======================
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# ======================
# 6. Training setup
# ======================
MODEL_DIR = os.path.join("models", "distilbert")

training_args = TrainingArguments(
    output_dir=MODEL_DIR,               # ✅ local folder
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="logs",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True
)

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy.compute(predictions=preds, references=labels)
    f1_score = f1.compute(predictions=preds, references=labels, average="macro")
    return {"accuracy": acc["accuracy"], "f1": f1_score["f1"]}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# ======================
# 7. Train
# ======================
trainer.train()

# ======================
# 8. Save model
# ======================
os.makedirs("models/distilbert", exist_ok=True)
trainer.save_model("models/distilbert")
tokenizer.save_pretrained("models/distilbert")

print("\n✅ Transformer model saved to models/distilbert/")
