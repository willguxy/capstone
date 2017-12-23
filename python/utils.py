"""
utility functions
"""

import numpy as np
import pandas as pd


def get_history(filename="2017-06-11", end_date="", delta_t=50, length=150, col="price"):
    data = pd.read_csv("../data/" + filename, parse_dates=["Time"])
    data["timestamp"] = data.Time
    data["price"] = data[col]
    data["time"] = np.arange(data.shape[0])
    data = data[data.Time <= end_date]
    if delta_t == 0:
        return data[-length:][["time", "price", "timestamp"]]
    return data[- delta_t -length: -delta_t][["time", "price", "timestamp"]]


def simulate_lppls(delta_t, length):
    A, B, C, tc, m, w, phi, sig = 8, -.015, .0015, 1080.01, .8, 9, 0.0, .03
    t = np.arange(int(tc) - delta_t - length, int(tc) - delta_t)
    LPPLS = A + \
            B * np.power(np.abs(tc - t), m) + \
            C * np.power(np.abs(tc - t), m) * np.cos(w * np.log(np.abs(tc - t)) + phi)

    return pd.DataFrame({
                "time": t,
                "price": np.exp(LPPLS + np.random.normal(0, sig, len(t))),
                "timestamp": [str(x) for x in t]
    })
