import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score

from preprocess import clean_text   # our preprocessing function

# ======================
# 1. Load dataset
# ======================
DATA_PATH = os.path.join("data", "reply_classification_dataset.csv")
df = pd.read_csv(DATA_PATH)

# Drop missing values and clean text
df.dropna(subset=["reply", "label"], inplace=True)
df["reply"] = df["reply"].apply(clean_text)

print("Class distribution:\n", df["label"].value_counts(), "\n")

# ======================
# 2. Train-test split
# ======================
if df["label"].value_counts().min() > 1:
    # stratify only if all classes have at least 2 samples
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )
else:
    print("⚠️ Some classes have <2 samples. Using random split without stratify.")
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42
    )

# ======================
# 3. Pipeline (TF-IDF + Logistic Regression)
# ======================
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

# Train
pipeline.fit(train_df["reply"], train_df["label"])

# ======================
# 4. Evaluate
# ======================
preds = pipeline.predict(test_df["reply"])
print("\nClassification Report:\n", classification_report(test_df["label"], preds))
print("Accuracy:", accuracy_score(test_df["label"], preds))
print("F1 Score (macro):", f1_score(test_df["label"], preds, average="macro"))

# ======================
# 5. Save model
# ======================
os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, os.path.join("models", "tfidf_logreg.joblib"))

print("\n✅ Baseline model saved to models/tfidf_logreg.joblib")
