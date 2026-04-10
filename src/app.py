"""
Smart Suspicious Message Detection Platform
Main Flask application — all API routes live here.

Run from project root:
    python app1.py
or
    python -m src.app   (if you want to run as module)
"""
from flask import (Flask, render_template, request, jsonify,
                   session, send_file)
import pickle, io, csv, threading, imaplib, email
from email.header import decode_header
from datetime import datetime

from .config import (MODEL_PATH, VECTORIZER_PATH, MODEL_INFO_PATH,
                     FAKE_THRESHOLD, SUSPICIOUS_LOW, SUSPICIOUS_HIGH)
from .utils        import load_pickle
from .preprocess   import clean_text
from .explain      import explain_prediction
from .ocr          import extract_text_from_image, OCR_AVAILABLE


# ── App bootstrap ──────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static",
)
app.secret_key = "smdp_secret_2025"

model     = load_pickle(MODEL_PATH)
vectorizer = load_pickle(VECTORIZER_PATH)
try:
    model_info = load_pickle(MODEL_INFO_PATH)
except Exception:
    model_info = {"accuracy": 0, "features": 0, "training_samples": 0}


# ── In-memory stores ───────────────────────────────────────────────────────────
prediction_history = []   # [{id, message, label, confidence, is_fake, ts}, …]

gmail_state = {
    "total": 0, "fake": 0, "suspicious": 0, "real": 0,
    "processing": False, "progress": 0,
    "messages": []
}

train_state = {
    "is_training": False, "progress": 0,
    "message": "", "last_accuracy": None
}


# ── Core prediction helper ─────────────────────────────────────────────────────
def run_prediction(raw_text: str) -> dict:
    """
    Returns a unified dict for any prediction call.
    Uses probability thresholds so not every message becomes Fake.
    """
    cleaned  = clean_text(raw_text)
    vec      = vectorizer.transform([cleaned])
    pred_raw = int(model.predict(vec)[0])

    if hasattr(model, "predict_proba"):
        proba      = model.predict_proba(vec)[0]
        real_prob  = float(proba[0])
        fake_prob  = float(proba[1])
    else:
        fake_prob = 1.0 if pred_raw == 1 else 0.0
        real_prob = 1.0 - fake_prob

    # ── Threshold logic ──────────────────────────────────────────────────────
    if fake_prob >= FAKE_THRESHOLD:
        label   = "Fake"
        is_fake = True
        badge   = "fake"
    elif fake_prob >= SUSPICIOUS_LOW:
        label   = "Suspicious"
        is_fake = False
        badge   = "suspicious"
    else:
        label   = "Real"
        is_fake = False
        badge   = "real"

    confidence  = fake_prob * 100 if is_fake else real_prob * 100
    explanation = explain_prediction(raw_text, 1 if is_fake else 0, fake_prob)

    return {
        "label":       label,
        "badge":       badge,
        "is_fake":     is_fake,
        "confidence":  round(confidence, 2),
        "fake_prob":   round(fake_prob * 100, 2),
        "real_prob":   round(real_prob * 100, 2),
        "explanation": explanation,
    }


def _save_history(raw_text: str, result: dict):
    entry = {
        "id":         len(prediction_history) + 1,
        "message":    raw_text[:120] + ("…" if len(raw_text) > 120 else ""),
        "label":      result["label"],
        "confidence": result["confidence"],
        "is_fake":    result["is_fake"],
        "fake_prob":  result["fake_prob"],
        "real_prob":  result["real_prob"],
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    prediction_history.append(entry)
    if len(prediction_history) > 1000:
        prediction_history.pop(0)
    return entry


def _dashboard_rows():
    rows = [
        {**entry, "source": "Direct Scan"}
        for entry in prediction_history
    ]

    for msg in gmail_state["messages"]:
        rows.append({
            "id": f"gmail-{msg['id']}",
            "message": msg["message"],
            "label": msg["label"],
            "confidence": msg["confidence_value"],
            "is_fake": msg["label"] == "Fake",
            "fake_prob": msg["fake_prob"],
            "real_prob": msg["real_prob"],
            "timestamp": msg["timestamp"],
            "source": "Gmail Scan",
        })

    rows.sort(key=lambda entry: entry["timestamp"], reverse=True)
    return rows


# ── Pages ──────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


# ── Message predict ────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    try:
        message = request.form.get("message", "").strip()
        if not message:
            return jsonify({"error": "Please enter a message"}), 400
        result = run_prediction(message)
        _save_history(message, result)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Image OCR predict ──────────────────────────────────────────────────────────
@app.route("/predict-image", methods=["POST"])
def predict_image():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file uploaded"}), 400
        img_file = request.files["image"]
        if img_file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        extracted_text = extract_text_from_image(img_file)
        result         = run_prediction(extracted_text)
        result["extracted_text"] = extracted_text
        _save_history(f"[IMAGE] {extracted_text}", result)
        return jsonify(result)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ocr-status")
def ocr_status():
    return jsonify({"available": OCR_AVAILABLE})


# ── Analytics data ─────────────────────────────────────────────────────────────
@app.route("/analytics")
def analytics():
    analytics_rows = _dashboard_rows()
    total      = len(analytics_rows)
    fake_c     = sum(1 for p in analytics_rows if p["label"] == "Fake")
    susp_c     = sum(1 for p in analytics_rows if p["label"] == "Suspicious")
    real_c     = total - fake_c - susp_c

    conf_buckets = {"0-40": 0, "40-60": 0, "60-80": 0, "80-100": 0}
    for p in analytics_rows:
        c = p["fake_prob"]
        if c < 40:     conf_buckets["0-40"]   += 1
        elif c < 60:   conf_buckets["40-60"]  += 1
        elif c < 80:   conf_buckets["60-80"]  += 1
        else:          conf_buckets["80-100"] += 1

    recent10 = analytics_rows[:10]

    return jsonify({
        "total": total, "fake": fake_c, "suspicious": susp_c, "real": real_c,
        "fake_pct": round(fake_c / total * 100 if total else 0, 1),
        "conf_buckets": conf_buckets,
        "recent": recent10,
        "source_counts": {
            "direct": len(prediction_history),
            "gmail": len(gmail_state["messages"]),
        },
        "model_accuracy":  round(model_info.get("accuracy", 0) * 100, 2),
        "model_features":  model_info.get("features", 0),
        "model_samples":   model_info.get("training_samples", 0),
    })


# ── History ────────────────────────────────────────────────────────────────────
@app.route("/history")
def get_history():
    return jsonify({"history": list(reversed(prediction_history[-50:]))})


@app.route("/clear-history", methods=["POST"])
def clear_history():
    prediction_history.clear()
    gmail_state.update({
        "total": 0,
        "fake": 0,
        "suspicious": 0,
        "real": 0,
        "processing": False,
        "progress": 0,
        "messages": [],
    })
    return jsonify({"success": True})


# ── CSV Export ─────────────────────────────────────────────────────────────────
def _make_csv(rows, filename_prefix):
    out = io.StringIO()
    w   = csv.writer(out)
    w.writerow(["ID", "Message Preview", "Label", "Confidence (%)",
                "Fake Prob (%)", "Real Prob (%)", "Timestamp"])
    for p in rows:
        w.writerow([p["id"], p["message"], p["label"],
                    p["confidence"], p["fake_prob"], p["real_prob"],
                    p["timestamp"]])
    out.seek(0)
    fname = f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return send_file(io.BytesIO(out.getvalue().encode("utf-8")),
                     mimetype="text/csv", as_attachment=True,
                     download_name=fname)


@app.route("/download-all-csv")
def download_all():
    return _make_csv(prediction_history, "all_results")


@app.route("/download-fake-csv")
def download_fake():
    return _make_csv([p for p in prediction_history if p["label"] == "Fake"],
                     "fake_only")


@app.route("/download-real-csv")
def download_real():
    return _make_csv([p for p in prediction_history if p["label"] == "Real"],
                     "real_only")


# ── Model info ─────────────────────────────────────────────────────────────────
@app.route("/model-info")
def get_model_info():
    return jsonify({
        "accuracy": round(model_info.get("accuracy", 0) * 100, 2),
        "features": model_info.get("features", 0),
        "training_samples": model_info.get("training_samples", 0),
    })


# ── Retrain from uploaded CSV ──────────────────────────────────────────────────
@app.route("/upload-train-data", methods=["POST"])
def upload_train_data():
    global model, vectorizer, model_info, train_state
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    if not f.filename.endswith(".csv"):
        return jsonify({"error": "Only .csv files accepted"}), 400
    try:
        import pandas as pd
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        from .preprocess import load_and_prepare_data
        from .utils import save_pickle
        import tempfile, os

        train_state.update({"is_training": True, "progress": 10,
                             "message": "Reading CSV…"})

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            f.save(tmp.name)
            tmp_path = tmp.name

        df = load_and_prepare_data(tmp_path)
        os.unlink(tmp_path)

        train_state.update({"progress": 40, "message": "Vectorising…"})
        X, y = df["message"].values, df["label"].values
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                               random_state=42, stratify=y)
        from .feature_engineering import build_vectorizer
        nv = build_vectorizer()
        Xtr_v = nv.fit_transform(Xtr)
        Xte_v = nv.transform(Xte)

        train_state.update({"progress": 70, "message": "Training…"})
        nm = LogisticRegression(max_iter=2000, class_weight="balanced",
                                random_state=42)
        nm.fit(Xtr_v, ytr)
        acc = float(accuracy_score(yte, nm.predict(Xte_v)))

        train_state.update({"progress": 90, "message": "Saving…"})
        import os as _os
        base = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        save_pickle(nm, _os.path.join(base, "model.pkl"))
        save_pickle(nv, _os.path.join(base, "vectorizer.pkl"))
        ni = {"accuracy": acc, "features": Xtr_v.shape[1],
              "training_samples": len(Xtr), "test_samples": len(Xte)}
        save_pickle(ni, _os.path.join(base, "model_info.pkl"))

        model      = nm
        vectorizer = nv
        model_info = ni
        train_state.update({"is_training": False, "progress": 100,
                             "message": "Done!", "last_accuracy": round(acc*100,2)})

        return jsonify({"success": True, "accuracy": round(acc*100,2),
                        "samples": len(df), "features": Xtr_v.shape[1]})
    except Exception as e:
        train_state["is_training"] = False
        return jsonify({"error": str(e)}), 500


@app.route("/training-status")
def training_status():
    return jsonify(train_state)


# ── Gmail ──────────────────────────────────────────────────────────────────────
def _decode_subject(s):
    if not s:
        return "No Subject"
    parts = []
    for part, enc in decode_header(s):
        parts.append(part.decode(enc or "utf-8", errors="ignore")
                     if isinstance(part, bytes) else str(part))
    return "".join(parts)


def _get_body(msg):
    body = ""
    try:
        if msg.is_multipart():
            for part in msg.walk():
                if (part.get_content_type() == "text/plain" and
                        "attachment" not in str(part.get("Content-Disposition"))):
                    p = part.get_payload(decode=True)
                    if p:
                        body = p.decode(errors="ignore"); break
        else:
            p = msg.get_payload(decode=True)
            if p:
                body = p.decode(errors="ignore")
    except Exception:
        pass
    return body.strip()


@app.route("/gmail-login", methods=["POST"])
def gmail_login():
    try:
        d   = request.get_json(silent=True) or {}
        em  = d.get("email", "").strip()
        pw  = d.get("password", "").strip()
        if not em or not pw:
            return jsonify({"error": "Email and App Password required"}), 400
        m = imaplib.IMAP4_SSL("imap.gmail.com")
        m.login(em, pw); m.logout()
        session["gmail_email"]    = em
        session["gmail_password"] = pw
        return jsonify({"success": True})
    except imaplib.IMAP4.error:
        return jsonify({"error": "Invalid credentials. Use a Gmail App Password."}), 401
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/gmail-logout", methods=["POST"])
def gmail_logout():
    session.pop("gmail_email", None)
    session.pop("gmail_password", None)
    gmail_state.update({
        "total": 0,
        "fake": 0,
        "suspicious": 0,
        "real": 0,
        "processing": False,
        "progress": 0,
        "messages": [],
    })
    return jsonify({"success": True})


@app.route("/check-gmail", methods=["POST"])
def check_gmail():
    em = session.get("gmail_email")
    pw = session.get("gmail_password")
    if not em or not pw:
        return jsonify({"error": "Not logged in to Gmail"}), 401
    threading.Thread(target=_process_gmail, args=(em, pw), daemon=True).start()
    return jsonify({"success": True})


def _process_gmail(em, pw):
    global gmail_state
    gmail_state.update({"total":0,"fake":0,"suspicious":0,"real":0,
                         "processing":True,"progress":0,"messages":[]})
    try:
        m = imaplib.IMAP4_SSL("imap.gmail.com")
        m.login(em, pw); m.select("INBOX")
        _, ids = m.search(None, "ALL")
        all_ids = ids[0].split()
        gmail_state["total"] = len(all_ids)
        batch = all_ids[-50:]
        for i, mid in enumerate(batch):
            try:
                _, data = m.fetch(mid, "(RFC822)")
                msg  = email.message_from_bytes(data[0][1])
                subj = _decode_subject(msg.get("Subject",""))
                frm  = msg.get("From","")
                body = _get_body(msg)
                res  = run_prediction(f"{subj} {body}".strip())
                gmail_state["messages"].append({
                    "id":       mid.decode() if isinstance(mid,bytes) else str(mid),
                    "from":     frm, "subject": subj,
                    "preview":  body[:140] + ("…" if len(body)>140 else ""),
                    "message":  f"{subj} {body[:160]}".strip()[:220],
                    "label":    res["label"],
                    "badge":    res["badge"],
                    "fake_prob": res["fake_prob"],
                    "real_prob": res["real_prob"],
                    "confidence_value": res["confidence"],
                    "confidence": f"{res['confidence']:.1f}%",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })
                if res["is_fake"]:
                    gmail_state["fake"] += 1
                elif res["label"] == "Suspicious":
                    gmail_state["suspicious"] += 1
                else:
                    gmail_state["real"] += 1
            except Exception:
                pass
            gmail_state["progress"] = int((i+1)/len(batch)*100)
        m.close(); m.logout()
    except Exception as e:
        print(f"Gmail error: {e}")
    gmail_state["processing"] = False


@app.route("/gmail-stats")
def gmail_stats():
    return jsonify(gmail_state)


@app.route("/delete-fake-messages", methods=["POST"])
def delete_fake():
    em = session.get("gmail_email")
    pw = session.get("gmail_password")
    if not em or not pw:
        return jsonify({"error": "Not logged in"}), 401
    ids = (request.get_json(silent=True) or {}).get("message_ids", [])
    if not ids:
        return jsonify({"error": "No IDs provided"}), 400
    try:
        m = imaplib.IMAP4_SSL("imap.gmail.com")
        m.login(em, pw); m.select("INBOX")
        cnt = 0
        for mid in ids:
            try: m.store(str(mid), "+FLAGS", "\\Deleted"); cnt += 1
            except Exception: pass
        m.expunge(); m.close(); m.logout()
        return jsonify({"success": True, "deleted": cnt})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/download-gmail-csv")
def download_gmail_csv():
    out = io.StringIO()
    w   = csv.writer(out)
    w.writerow(["From","Subject","Preview","Label","Confidence","Timestamp"])
    for x in gmail_state["messages"]:
        w.writerow([x["from"],x["subject"],x["preview"],
                    x["label"],x["confidence"],x["timestamp"]])
    out.seek(0)
    fname = f"gmail_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return send_file(io.BytesIO(out.getvalue().encode("utf-8")),
                     mimetype="text/csv", as_attachment=True,
                     download_name=fname)
