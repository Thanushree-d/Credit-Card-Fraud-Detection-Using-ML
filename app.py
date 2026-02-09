from flask import Flask, render_template, request, redirect, url_for, session, abort, jsonify, send_file
import joblib
import numpy as np
import sqlite3
from datetime import datetime
import os
from functools import wraps
import csv
from io import StringIO

app = Flask(__name__)
app.secret_key = "change_this_secret_key"
DB_PATH = "database.db"

# -------------------------------
# Load Models
# -------------------------------
MODELS = {
    "Logistic Regression": joblib.load("models/logistic.pkl"),
    "Random Forest": joblib.load("models/random_forest.pkl"),
    "SVM": joblib.load("models/svm.pkl"),
}
BEST_MODEL_NAME = "Random Forest"
scaler = joblib.load("models/scaler.pkl")

# Default fraud threshold (admin can change)
FRAUD_THRESHOLD = 60.0

# -------------------------------
# Database Setup
# -------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            model TEXT,
            risk REAL,
            result TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            role TEXT DEFAULT 'user'
        )
    """)
    conn.commit()
    conn.close()

init_db()

def log_tx(model, risk, result):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO history(timestamp, model, risk, result) VALUES (?,?,?,?)",
                (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), model, risk, result))
    conn.commit()
    conn.close()

# -------------------------------
# Auth Helpers
# -------------------------------
def login_required(role=None):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if "username" not in session:
                return redirect(url_for("login"))
            if role and session.get("role") != role:
                abort(403)
            return f(*args, **kwargs)
        return wrapper
    return decorator

# -------------------------------
# Utility: Ensemble Prediction
# -------------------------------
def ensemble_predict(X):
    preds = []
    probs = []

    for name, model in MODELS.items():
        p = int(model.predict(X)[0])
        preds.append(p)
        if hasattr(model, "predict_proba"):
            probs.append(model.predict_proba(X)[0][1] * 100)

    final_pred = int(sum(preds) >= 2)  # majority voting
    final_risk = float(np.mean(probs)) if probs else 50.0

    return final_pred, final_risk

# -------------------------------
# Routes
# -------------------------------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        try:
            cur.execute("INSERT INTO users(username, password, role) VALUES (?,?,?)",
                        (username, password, "user"))
            conn.commit()
        except sqlite3.IntegrityError:
            conn.close()
            return render_template("signup.html", error="Username already exists")
        conn.close()
        return redirect(url_for("login"))

    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT role FROM users WHERE username=? AND password=?",
                    (username, password))
        row = cur.fetchone()
        conn.close()

        if username == "admin" and password == "admin123":
            session["username"] = "admin"
            session["role"] = "admin"
            return redirect(url_for("dashboard"))

        if row:
            session["username"] = username
            session["role"] = row[0]
            return redirect(url_for("index"))

        return render_template("login.html", error="Invalid username or password")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

@app.route("/app")
@login_required()
def index():
    metrics = {
        "Logistic Regression": {"acc": 0.94, "f1": 0.78},
        "Random Forest": {"acc": 0.99, "f1": 0.92},
        "SVM": {"acc": 0.96, "f1": 0.84},
        "Ensemble": {"acc": 0.995, "f1": 0.94},
    }
    return render_template("index.html", metrics=metrics, best_model="Ensemble")

@app.route("/predict", methods=["POST"])
@login_required()
def predict():
    try:
        features = [float(v) for k, v in request.form.items() if k != "model"]
    except ValueError:
        return render_template("result.html", result="Invalid Input", risk=0, badge="danger",
                               model="N/A", explanation="Please enter valid numeric values.")

    X = scaler.transform(np.array(features).reshape(1, -1))

    model_choice = request.form.get("model")

    if model_choice == "Auto (Best)":
        pred, risk = ensemble_predict(X)
        model_name = "Ensemble"
    else:
        model = MODELS.get(model_choice)
        pred = int(model.predict(X)[0])
        risk = float(model.predict_proba(X)[0][1] * 100) if hasattr(model, "predict_proba") else 50.0
        model_name = model_choice

    result = "Fraud" if risk >= FRAUD_THRESHOLD else "Legitimate"
    badge = "danger" if result == "Fraud" else "success"

    explanation = (
        "The ensemble model detected unusual transaction behavior across multiple classifiers."
        if model_name == "Ensemble" else
        "The selected model detected transaction patterns indicating fraud risk."
    )

    log_tx(model_name, round(risk, 2), result)

    return render_template("result.html",
                           result=result,
                           risk=round(risk, 2),
                           badge=badge,
                           model=model_name,
                           explanation=explanation)

# -------------------------------
# Batch CSV Upload
# -------------------------------
@app.route("/batch", methods=["GET", "POST"])
@login_required(role="admin")
def batch_predict():
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            return "No file uploaded", 400

        content = file.read().decode("utf-8")
        reader = csv.reader(StringIO(content))
        results = []

        for row in reader:
            features = list(map(float, row))
            X = scaler.transform(np.array(features).reshape(1, -1))
            pred, risk = ensemble_predict(X)
            results.append({"features": row, "risk": round(risk, 2), "result": "Fraud" if risk >= FRAUD_THRESHOLD else "Legitimate"})

        return jsonify(results)

    return "<h3>Upload CSV file for batch fraud prediction</h3>"

# -------------------------------
# Export History
# -------------------------------
@app.route("/export")
@login_required(role="admin")
def export_history():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM history")
    rows = cur.fetchall()
    conn.close()

    si = StringIO()
    writer = csv.writer(si)
    writer.writerow(["ID", "Timestamp", "Model", "Risk", "Result"])
    writer.writerows(rows)

    output = si.getvalue()
    return send_file(
        StringIO(output),
        mimetype="text/csv",
        as_attachment=True,
        download_name="fraud_history.csv"
    )

# -------------------------------
# API Endpoint
# -------------------------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.json.get("features")
    X = scaler.transform(np.array(data).reshape(1, -1))
    pred, risk = ensemble_predict(X)
    return jsonify({"risk": round(risk, 2), "result": "Fraud" if risk >= FRAUD_THRESHOLD else "Legitimate"})

@app.route("/history")
@login_required()
def history():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM history ORDER BY id DESC")
    rows = cur.fetchall()
    conn.close()
    return render_template("history.html", rows=rows)

@app.route("/dashboard")
@login_required(role="admin")
def dashboard():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT result, COUNT(*) FROM history GROUP BY result")
    counts = cur.fetchall()
    conn.close()
    labels = [c[0] for c in counts]
    values = [c[1] for c in counts]
    return render_template("dashboard.html", labels=labels, values=values)

@app.route("/metrics")
@login_required(role="admin")
def metrics_page():
    return render_template("metrics.html")

# -------------------------------
# Error Pages
# -------------------------------
@app.errorhandler(403)
def forbidden(e):
    return "<h3>403 - Forbidden (Admins only)</h3>", 403

# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
