import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from features import FEATURES, features_from_window

BASE = Path(__file__).resolve().parent  # Vercel pe cwd alag ho sakta hai, isliye absolute paths


def locate(name, *folders):
    """File pehle folder (model/, data/) me dhundho, nahi mili to root me. Dono layout chalte hain."""
    for f in folders + ("",):
        p = BASE / f / name
        if p.exists():
            return p
    raise FileNotFoundError(f"{name} nahi mili. Check karo ki file repo me hai (folders: {folders})")


template_dir = BASE / "templates" if (BASE / "templates" / "index.html").exists() else BASE
app = Flask(__name__, template_folder=str(template_dir))

bundle = joblib.load(locate("oil_model.joblib", "model"))
model = bundle["model"]
metrics = json.load(open(locate("metrics.json", "model")))
data = pd.read_csv(locate("clean_oil_data.csv", "data"), parse_dates=["Date"])
test_preds = pd.read_csv(locate("test_predictions.csv", "model"), parse_dates=["Date"])
prices = data["Price"].values


def next_price(window):
    f = pd.DataFrame([features_from_window(window)])[FEATURES]
    return float(window[-1] * (1 + model.predict(f)[0]))


@app.route("/")
def index():
    stats = {
        "latest": float(prices[-1]),
        "latest_date": data["Date"].iloc[-1].strftime("%d %b %Y"),
        "high": float(data["High"].max()),
        "low": float(data["Low"].min()),
        "avg": float(prices.mean()),
        "days": int(len(data)),
        "start": data["Date"].iloc[0].strftime("%d %b %Y"),
    }
    return render_template("index.html", stats=stats, m=metrics,
                           last5=[round(float(p), 2) for p in prices[-5:]])


@app.route("/api/history")
def history():
    ma20 = data["Price"].rolling(20).mean()
    return jsonify({
        "labels": data["Date"].dt.strftime("%d %b %y").tolist(),
        "price": data["Price"].tolist(),
        "ma20": [None if np.isnan(v) else round(float(v), 2) for v in ma20],
    })


@app.route("/api/forecast")
def forecast():
    n = max(1, min(int(request.args.get("days", 7)), 30))
    w, vals = list(prices[-5:]), []
    for _ in range(n):
        p = next_price(w[-5:])
        vals.append(p)
        w.append(p)
    dates = pd.bdate_range(data["Date"].iloc[-1] + pd.Timedelta(days=1), periods=n)  # trading days
    band = [1.96 * metrics["resid_std"] * np.sqrt(h + 1) for h in range(n)]
    tail = data.tail(30)
    return jsonify({
        "labels": dates.strftime("%d %b %y").tolist(),
        "values": [round(v, 2) for v in vals],
        "lower": [round(v - b, 2) for v, b in zip(vals, band)],
        "upper": [round(v + b, 2) for v, b in zip(vals, band)],
        "history": {"labels": tail["Date"].dt.strftime("%d %b %y").tolist(), "values": tail["Price"].tolist()},
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        w = [float(x) for x in request.get_json()["prices"]]
        if len(w) != 5 or any(x <= 0 for x in w):
            raise ValueError("5 positive prices chahiye (purani -> nayi)")
        p = next_price(w)
        return jsonify({"prediction": round(p, 2), "change": round(p - w[-1], 2),
                        "change_pct": round((p / w[-1] - 1) * 100, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/performance")
def performance():
    return jsonify({
        "labels": test_preds["Date"].dt.strftime("%d %b %y").tolist(),
        "actual": test_preds["Actual"].round(2).tolist(),
        "predicted": test_preds["Predicted"].round(2).tolist(),
    })


if __name__ == "__main__":
    app.run(debug=True)
