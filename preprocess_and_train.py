"""
IMDB Sentiment Analysis - Preprocessing & Model Training Pipeline
Covers: text cleaning, TF-IDF vectorization, Logistic Regression, evaluation
"""

import os
import re
import string
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

warnings.filterwarnings("ignore")

# ── Download NLTK data ────────────────────────────────────────────────────────
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── 1. Load Dataset ───────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading IMDB Dataset")
print("=" * 60)

df = pd.read_csv("IMDB Dataset.csv")
print(f"  Dataset shape : {df.shape}")
print(f"  Columns       : {df.columns.tolist()}")
print(f"  Sentiment dist:\n{df['sentiment'].value_counts().to_string()}")

# ── 2. Text Cleaning ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Text Preprocessing")
print("=" * 60)

STOP_WORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """
    Pipeline:
      1. Lowercase
      2. Remove HTML tags
      3. Remove URLs
      4. Remove punctuation & digits
      5. Tokenize
      6. Remove stopwords
      7. Rejoin tokens
    """
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)           # strip HTML
    text = re.sub(r"http\S+|www\S+", " ", text)  # strip URLs
    text = re.sub(r"[^a-z\s]", " ", text)        # keep only letters
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)


print("  Cleaning reviews (this may take ~30 s)…")
df["clean_review"] = df["review"].apply(clean_text)
df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})

# Show sample
print(f"\n  Original  : {df['review'].iloc[0][:120]}…")
print(f"  Cleaned   : {df['clean_review'].iloc[0][:120]}…")

# ── 3. TF-IDF Vectorization ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: TF-IDF Vectorization")
print("=" * 60)

X = df["clean_review"]
y = df["label"]

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train size : {len(X_train_raw):,}")
print(f"  Test  size : {len(X_test_raw):,}")

vectorizer = TfidfVectorizer(
    max_features=50_000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)
X_train = vectorizer.fit_transform(X_train_raw)
X_test  = vectorizer.transform(X_test_raw)
print(f"  Feature matrix shape (train): {X_train.shape}")
print(f"  Feature matrix shape (test) : {X_test.shape}")

# ── 4. Train Logistic Regression ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Training Logistic Regression")
print("=" * 60)

model = LogisticRegression(
    max_iter=1000,
    C=1.0,
    solver="lbfgs",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("  Model trained successfully.")

# ── 5. Evaluation ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Evaluation Metrics")
print("=" * 60)

y_pred = model.predict(X_test)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)

print(f"  Accuracy  : {accuracy:.4f}")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1-Score  : {f1:.4f}")
print("\n  Full Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

# Save metrics to file for report card
metrics = {
    "accuracy": round(accuracy, 4),
    "precision": round(precision, 4),
    "recall": round(recall, 4),
    "f1_score": round(f1, 4),
    "train_size": len(X_train_raw),
    "test_size": len(X_test_raw),
    "total_samples": len(df),
    "vocab_size": len(vectorizer.vocabulary_),
}
import json
os.makedirs("static", exist_ok=True)
with open("static/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("\n  Metrics saved -> static/metrics.json")

# ── 6. Confusion Matrix Chart ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Generating Charts")
print("=" * 60)

cm = confusion_matrix(y_test, y_pred)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Sentiment Analysis – Evaluation Visualizations", fontsize=14, fontweight="bold")

# Confusion matrix heatmap
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"],
    ax=axes[0]
)
axes[0].set_title("Confusion Matrix")
axes[0].set_xlabel("Predicted Label")
axes[0].set_ylabel("True Label")

# Sentiment distribution bar chart
sentiment_counts = df["sentiment"].value_counts()
bars = axes[1].bar(
    sentiment_counts.index,
    sentiment_counts.values,
    color=["#e74c3c", "#2ecc71"],
    edgecolor="black",
    width=0.5
)
for bar, val in zip(bars, sentiment_counts.values):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                 f"{val:,}", ha="center", va="bottom", fontweight="bold")
axes[1].set_title("Sentiment Distribution in Dataset")
axes[1].set_xlabel("Sentiment")
axes[1].set_ylabel("Count")
axes[1].set_ylim(0, max(sentiment_counts.values) * 1.15)

plt.tight_layout()
plt.savefig("static/evaluation_charts.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Charts saved -> static/evaluation_charts.png")

# ── 7. Save Model & Vectorizer ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Saving Model Artifacts")
print("=" * 60)

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("  Model saved     -> model.pkl")
print("  Vectorizer saved -> vectorizer.pkl")

print("\nPipeline complete. Run 'python app.py' to start the API server.")
