# src/evaluate.py
import joblib
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from preprocess import load_and_clean
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score, classification_report
from pathlib import Path

# Resolve project root and dataset/model paths reliably
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "reply_classification_dataset.csv"
MODEL_DIR = PROJECT_ROOT / "models"
DISTILBERT_DIR = MODEL_DIR / "distilbert"

# Load data
df = load_and_clean(str(DATA_PATH))
train_df, test_df = df, df  # Evaluate on all or do split again
texts = test_df['reply'].tolist()
labels = test_df['label'].tolist()

# Baseline
baseline = joblib.load(str(MODEL_DIR / "tfidf_logreg.joblib"))
pred_baseline = baseline.predict(texts)
print("Baseline Evaluation:")
print("Accuracy:", accuracy_score(labels, pred_baseline))
print("F1 (macro):", f1_score(labels, pred_baseline, average='macro'))
print(classification_report(labels, pred_baseline))

# Transformer
tokenizer = AutoTokenizer.from_pretrained(str(DISTILBERT_DIR))
model = AutoModelForSequenceClassification.from_pretrained(str(DISTILBERT_DIR))
id2label = model.config.id2label

preds = []
for t in texts:
    inputs = tokenizer(t, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_id = int(torch.argmax(logits, dim=1))
    preds.append(id2label[pred_id])

print("\nTransformer Evaluation:")
print("Accuracy:", accuracy_score(labels, preds))
print("F1 (macro):", f1_score(labels, preds, average='macro'))
print(classification_report(labels, preds))
