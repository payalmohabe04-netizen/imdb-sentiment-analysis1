"""
End-to-end validation of the full sentiment analysis pipeline.
Run after 'python app.py' is already started.
"""

import urllib.request
import urllib.error
import json
import sys

BASE = "http://127.0.0.1:5000"
passed = 0
failed = 0


def get(path):
    return urllib.request.urlopen(BASE + path)


def post(path, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        BASE + path, data=data,
        headers={"Content-Type": "application/json"}
    )
    try:
        resp = urllib.request.urlopen(req)
        return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  [PASS] {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name}  {detail}")
        failed += 1


print("=" * 60)
print("  FULL PIPELINE END-TO-END VALIDATION")
print("=" * 60)

# ── TEST 1: Health Check ──────────────────────────────────────────
print("\nTEST 1: Health Check")
resp = get("/health")
data = json.loads(resp.read())
check("HTTP 200", resp.status == 200)
check("status = ok", data.get("status") == "ok")
check("model field present", "model" in data)
print(f"  Response: {json.dumps(data)}")

# ── TEST 2: Metrics Endpoint ──────────────────────────────────────
print("\nTEST 2: Metrics Endpoint")
resp = get("/metrics")
m = json.loads(resp.read())
check("HTTP 200", resp.status == 200)
check("accuracy > 0.88", m.get("accuracy", 0) > 0.88)
check("f1_score > 0.88", m.get("f1_score", 0) > 0.88)
check("precision > 0.88", m.get("precision", 0) > 0.88)
check("recall > 0.88", m.get("recall", 0) > 0.88)
check("train_size = 40000", m.get("train_size") == 40000)
check("test_size  = 10000", m.get("test_size") == 10000)
check("total_samples = 50000", m.get("total_samples") == 50000)
print(
    f"  Accuracy={m['accuracy']}, Precision={m['precision']}, "
    f"Recall={m['recall']}, F1={m['f1_score']}"
)

# ── TEST 3: Positive Review ───────────────────────────────────────
print("\nTEST 3: Positive Review Prediction")
status, data = post("/predict", {
    "review": (
        "This movie was absolutely brilliant. Outstanding acting, "
        "story, and direction. A true masterpiece of cinema."
    )
})
check("HTTP 200", status == 200)
check("sentiment = positive", data.get("sentiment") == "positive")
check("label = 1", data.get("label") == 1)
check("confidence > 0.70", data.get("confidence", 0) > 0.70)
check("review_preview present", "review_preview" in data)
print(f"  sentiment={data['sentiment']}, confidence={data['confidence']}")

# ── TEST 4: Negative Review ───────────────────────────────────────
print("\nTEST 4: Negative Review Prediction")
status, data = post("/predict", {
    "review": (
        "Terrible film. The plot made no sense, acting was wooden, "
        "I almost fell asleep. Complete waste of money and time."
    )
})
check("HTTP 200", status == 200)
check("sentiment = negative", data.get("sentiment") == "negative")
check("label = 0", data.get("label") == 0)
check("confidence > 0.70", data.get("confidence", 0) > 0.70)
print(f"  sentiment={data['sentiment']}, confidence={data['confidence']}")

# ── TEST 5: Edge Case - Empty Review (expect 400) ─────────────────
print("\nTEST 5: Edge Case - Empty Review (expects HTTP 400)")
status, data = post("/predict", {"review": ""})
check("HTTP 400 returned", status == 400)
check("error field in response", "error" in data)
print(f"  Response: {json.dumps(data)}")

# ── TEST 6: Mixed/Ambiguous Review ───────────────────────────────
print("\nTEST 6: Mixed/Ambiguous Review")
status, data = post("/predict", {
    "review": (
        "The movie had great visuals but the story was weak "
        "and the ending was disappointing."
    )
})
check("HTTP 200", status == 200)
check("sentiment field present", "sentiment" in data)
check("confidence field present", "confidence" in data)
check("valid sentiment value", data.get("sentiment") in ["positive", "negative"])
print(f"  sentiment={data['sentiment']}, confidence={data['confidence']}")

# ── TEST 7: Long Review ───────────────────────────────────────────
print("\nTEST 7: Long Review Input")
long_review = (
    "I watched this film last night with my family and we were all "
    "blown away by the sheer quality of the production. Every scene "
    "was crafted with care, the dialogue was sharp and witty, and "
    "the performances were nothing short of extraordinary. The director "
    "clearly has a unique vision and the cinematography was breathtaking. "
    "I have not felt this moved by a film in years. Highly recommended "
    "to anyone who appreciates true artistry in filmmaking."
) * 3
status, data = post("/predict", {"review": long_review})
check("HTTP 200 for long input", status == 200)
check("sentiment = positive", data.get("sentiment") == "positive")
check("review_preview truncated to 200 chars",
      len(data.get("review_preview", "")) <= 200)
print(f"  sentiment={data['sentiment']}, confidence={data['confidence']}")

# ── TEST 8: Frontend HTML Pages ───────────────────────────────────
print("\nTEST 8: Frontend HTML Pages")
for path, label in [("/", "index"), ("/report", "report")]:
    resp = get(path)
    body = resp.read().decode()
    check(f"GET {path} returns HTTP 200", resp.status == 200)
    check(f"{label}.html contains DOCTYPE", "<!DOCTYPE html>" in body)
    check(f"{label}.html has <title>", "<title>" in body)

# ── TEST 9: Artifacts on Disk ────────────────────────────────────
print("\nTEST 9: Required Artifacts on Disk")
import os
artifacts = [
    "model.pkl",
    "vectorizer.pkl",
    "static/metrics.json",
    "static/evaluation_charts.png",
    "templates/index.html",
    "templates/report.html",
    "app.py",
    "preprocess_and_train.py",
    "README.md",
]
for f in artifacts:
    exists = os.path.exists(f)
    size = os.path.getsize(f) if exists else 0
    check(f"{f} exists ({size:,} bytes)", exists and size > 0)

# ── Summary ───────────────────────────────────────────────────────
total = passed + failed
print()
print("=" * 60)
print(f"  RESULTS: {passed}/{total} tests passed")
if failed == 0:
    print("  ALL TESTS PASSED - Pipeline fully validated!")
else:
    print(f"  {failed} test(s) FAILED")
print("=" * 60)

sys.exit(0 if failed == 0 else 1)
