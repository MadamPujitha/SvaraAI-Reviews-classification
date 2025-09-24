# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import joblib
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

MODEL_DIR = "models"
TRANSFORMER_DIR = os.path.join(MODEL_DIR, "distilbert")
BASELINE_FILE = os.path.join(MODEL_DIR, "tfidf_logreg.joblib")

app = FastAPI(title="Reply Classifier")

class InputText(BaseModel):
    text: str

# Load models
transformer = None
tokenizer = None
id2label = None
if os.path.isdir(TRANSFORMER_DIR):
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_DIR)
    transformer = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_DIR)
    id2label = transformer.config.id2label

baseline = None
if os.path.exists(BASELINE_FILE):
    baseline = joblib.load(BASELINE_FILE)

if transformer is None and baseline is None:
    raise RuntimeError("No model found.")

@app.post("/predict")
def predict(inp: InputText):
    text = inp.text
    if not text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    # Transformer
    if transformer is not None:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            logits = transformer(**inputs).logits.detach().cpu().numpy()[0]
        probs = softmax(logits)
        pred_id = int(np.argmax(probs))
        return {"label": id2label[pred_id], "confidence": float(probs[pred_id])}

    # Baseline
    probs = baseline.predict_proba([text])[0]
    classes = baseline.classes_
    idx = int(np.argmax(probs))
    return {"label": classes[idx], "confidence": float(probs[idx])}
