import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import build_dataset


def make_models():
    return {
        "Ridge Regression": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=4, min_samples_leaf=5, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05,
                                                       subsample=0.8, random_state=42),
    }


def score(p_today, p_actual, p_pred, with_direction=True):
    err = p_actual - p_pred
    out = {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE": float(np.mean(np.abs(err) / p_actual) * 100),
        "R2": float(r2_score(p_actual, p_pred)),
        "Direction": None,
    }
    if with_direction:
        out["Direction"] = float(np.mean(np.sign(p_pred - p_today) == np.sign(p_actual - p_today)) * 100)
    return out


def cv_rmse(model, X, y, p0):
    """TimeSeriesSplit: hamesha past pe train, future pe validate"""
    errs = []
    for tr, va in TimeSeriesSplit(n_splits=4).split(X):
        m = clone(model).fit(X.iloc[tr], y[tr])
        pred_price = p0[va] * (1 + m.predict(X.iloc[va]))
        act_price = p0[va] * (1 + y[va])
        errs.append(np.sqrt(np.mean((act_price - pred_price) ** 2)))
    return float(np.mean(errs))


def cv_rmse_baseline(y, p0, n_splits=4, X=None):
    errs = []
    for tr, va in TimeSeriesSplit(n_splits=n_splits).split(X):
        errs.append(np.sqrt(np.mean((p0[va] * y[va]) ** 2)))  # actual - p0 = p0*y
    return float(np.mean(errs))


def main():
    df = pd.read_csv("data/clean_oil_data.csv", parse_dates=["Date"])
    prices, dates = df["Price"].values, df["Date"].values

    X, y, idx = build_dataset(prices)
    p0, p_next, d_next = prices[idx], prices[idx + 1], dates[idx + 1]

    split = int(len(X) * 0.8)  # time-based split (shuffle nahi)
    Xtr, Xte, ytr, yte = X.iloc[:split], X.iloc[split:], y[:split], y[split:]
    p0_tr, p0_te, pn_te, d_te = p0[:split], p0[split:], p_next[split:], d_next[split:]

    results, fitted = [], {}
    for name, model in make_models().items():
        cv = cv_rmse(model, Xtr, ytr, p0_tr)
        m = clone(model).fit(Xtr, ytr)
        pred_price = p0_te * (1 + m.predict(Xte))
        res = {"name": name, "CV_RMSE": cv, **score(p0_te, pn_te, pred_price)}
        results.append(res)
        fitted[name] = pred_price
        print(f"{name:18s} CV_RMSE={cv:.3f}  Test MAE={res['MAE']:.3f}  RMSE={res['RMSE']:.3f}  Dir={res['Direction']:.1f}%")

    baseline = {"name": "Naive baseline (kal = aaj)", "CV_RMSE": cv_rmse_baseline(ytr, p0_tr, X=Xtr),
                **score(p0_te, pn_te, p0_te, with_direction=False)}
    print(f"{'Naive baseline':18s} CV_RMSE={baseline['CV_RMSE']:.3f}  Test MAE={baseline['MAE']:.3f}  RMSE={baseline['RMSE']:.3f}")

    best = min(results, key=lambda r: r["CV_RMSE"])["name"]  # selection CV se, test se nahi
    best_pred = fitted[best]
    resid_std = float(np.std(pn_te - best_pred))

    final = clone(make_models()[best]).fit(X, y)  # deployment ke liye poore data pe refit
    Path("model").mkdir(exist_ok=True)
    joblib.dump({"model": final, "name": best}, "model/oil_model.joblib")

    pd.DataFrame({"Date": pd.to_datetime(d_te), "Actual": pn_te, "Predicted": best_pred, "Naive": p0_te}) \
        .to_csv("model/test_predictions.csv", index=False)

    json.dump({
        "models": results, "baseline": baseline, "best": best, "resid_std": resid_std,
        "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
        "test_start": str(pd.to_datetime(d_te[0]).date()), "test_end": str(pd.to_datetime(d_te[-1]).date()),
    }, open("model/metrics.json", "w"), indent=2)
    print("Best model (CV se):", best)


if __name__ == "__main__":
    main()
