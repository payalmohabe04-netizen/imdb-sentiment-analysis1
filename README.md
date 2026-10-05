# IMDB Sentiment Analysis — Project Documentation

## Overview
End-to-end NLP pipeline that trains a sentiment classifier on 50,000 IMDB movie reviews and serves predictions through a REST API with a browser-based UI.

---

## Project Structure
```
sentiment_analysis/
├── IMDB Dataset.csv            # Raw dataset (50,000 reviews)
├── preprocess_and_train.py     # Full ML pipeline: clean → vectorize → train → evaluate → save
├── app.py                      # Flask REST API server
├── model.pkl                   # Trained Logistic Regression model
├── vectorizer.pkl              # Fitted TF-IDF vectorizer
├── templates/
│   ├── index.html              # Frontend: review input & sentiment output
│   └── report.html             # Evaluation report card (metrics + charts)
├── static/
│   ├── metrics.json            # Saved evaluation metrics
│   └── evaluation_charts.png   # Confusion matrix + sentiment distribution chart
└── README.md                   # This file
```

---

## 1. Data Preprocessing

### Dataset
- **Source:** IMDB Dataset (50,000 labeled movie reviews)
- **Classes:** Positive (25,000) · Negative (25,000) — perfectly balanced
- **Columns:** `review` (raw text), `sentiment` (positive/negative)

### Text Cleaning Steps (applied in order)
| Step | Operation | Example |
|------|-----------|---------|
| 1 | **Lowercase** | `"GREAT Movie"` → `"great movie"` |
| 2 | **Remove HTML tags** | `"great<br/>"` → `"great "` |
| 3 | **Remove URLs** | `"see http://imdb.com"` → `"see "` |
| 4 | **Remove non-alpha chars** | `"can't!"` → `"cant "` |
| 5 | **Tokenize** | `"great movie"` → `["great", "movie"]` |
| 6 | **Remove stopwords** | drops `"the"`, `"is"`, `"a"`, etc. (NLTK English) |
| 7 | **Filter short tokens** | drops single-character tokens |
| 8 | **Rejoin** | `["great", "movie"]` → `"great movie"` |

---

## 2. Feature Engineering — TF-IDF Vectorization

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `max_features` | 50,000 | Top 50k most informative terms |
| `ngram_range` | (1, 2) | Unigrams + bigrams (captures phrases like "not good") |
| `min_df` | 2 | Ignore terms appearing in fewer than 2 documents |
| `sublinear_tf` | True | Apply `1 + log(tf)` scaling to reduce impact of high-frequency terms |

- **Train–Test Split:** 80% train (40,000) / 20% test (10,000), stratified

---

## 3. Model — Logistic Regression

| Parameter | Value |
|-----------|-------|
| Solver | `lbfgs` |
| Regularization (C) | 1.0 |
| Max iterations | 1,000 |
| Random state | 42 |

**Why Logistic Regression?**
- Efficient and interpretable for high-dimensional sparse text features
- Competitive accuracy with TF-IDF for binary text classification
- Fast inference — suitable for real-time API serving

---

## 4. Evaluation Results

| Metric | Score |
|--------|-------|
| **Accuracy** | ~89–90% |
| **Precision** | ~89–90% |
| **Recall** | ~89–90% |
| **F1-Score** | ~89–90% |

*Exact values written to `static/metrics.json` after training.*

### Confusion Matrix
Visualized as a heatmap (saved to `static/evaluation_charts.png`):
- Rows = True labels (Negative / Positive)
- Columns = Predicted labels (Negative / Positive)

---

## 5. REST API

### Start the server
```bash
python app.py
```
Server runs at `http://127.0.0.1:5000`

### Endpoints

#### `POST /predict`
**Request body:**
```json
{ "review": "This movie was absolutely fantastic!" }
```
**Response:**
```json
{
  "sentiment":      "positive",
  "confidence":     0.9741,
  "label":          1,
  "review_preview": "This movie was absolutely fantastic!"
}
```

#### `GET /metrics`
Returns saved evaluation metrics:
```json
{
  "accuracy": 0.8997,
  "precision": 0.9012,
  "recall": 0.8978,
  "f1_score": 0.8995,
  "train_size": 40000,
  "test_size": 10000,
  "total_samples": 50000,
  "vocab_size": 47382
}
```

#### `GET /health`
```json
{ "status": "ok", "model": "LogisticRegression", "vectorizer": "TF-IDF" }
```

---

## 6. Frontend

### Sentiment Analyzer (`/`)
- Text area for review input
- Quick-fill example buttons (positive / negative)
- Keyboard shortcut: `Ctrl+Enter` to analyze
- Displays: sentiment label, confidence percentage, animated confidence bar

### Evaluation Report (`/report`)
- Live metric cards (Accuracy, Precision, Recall, F1)
- Animated horizontal progress bars per metric
- Confusion matrix + sentiment distribution chart image
- Dataset & model parameter reference table

---

## 7. How to Run

### Step 1 — Train the model
```bash
python preprocess_and_train.py
```
This will:
- Clean and preprocess all 50,000 reviews
- Fit TF-IDF vectorizer on training set
- Train Logistic Regression classifier
- Print evaluation report
- Save `model.pkl`, `vectorizer.pkl`, `static/metrics.json`, `static/evaluation_charts.png`

### Step 2 — Start the API
```bash
python app.py
```

### Step 3 — Open the UI
- Sentiment Analyzer: http://127.0.0.1:5000/
- Evaluation Report:  http://127.0.0.1:5000/report

---

## 8. Dependencies
```
pandas, numpy, scikit-learn, nltk, flask, flask-cors, matplotlib, seaborn, joblib
```
Install with:
```bash
pip install pandas numpy scikit-learn flask flask-cors nltk matplotlib seaborn joblib
```
