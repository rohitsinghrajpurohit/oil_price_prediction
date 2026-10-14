"""Features sirf pichle 5 closing prices se bante hain.
Isse web app me user sirf 5 prices daal ke bhi prediction le sakta hai,
aur multi-day forecast (recursive) bhi ban sakta hai.
"""
import numpy as np
import pandas as pd

FEATURES = ["ret1", "ret2", "ret3", "ret4", "ma3_ratio", "ma5_ratio", "vol4"]


def features_from_window(w):
    """w = last 5 closing prices, oldest -> newest"""
    w = np.asarray(w, dtype=float)
    p0 = w[-1]
    rets = w[1:] / w[:-1] - 1  # 4 daily returns
    return {
        "ret1": rets[-1], "ret2": rets[-2], "ret3": rets[-3], "ret4": rets[-4],
        "ma3_ratio": w[-3:].mean() / p0 - 1,
        "ma5_ratio": w.mean() / p0 - 1,
        "vol4": rets.std(),
    }


def build_dataset(prices):
    """Har din t ke liye: features (t tak ke 5 prices) -> target (agle din ka return)"""
    prices = np.asarray(prices, dtype=float)
    rows, y, idx = [], [], []
    for t in range(4, len(prices) - 1):
        rows.append(features_from_window(prices[t - 4: t + 1]))
        y.append(prices[t + 1] / prices[t] - 1)
        idx.append(t)
    return pd.DataFrame(rows)[FEATURES], np.array(y), np.array(idx)
