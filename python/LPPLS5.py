# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 23:05:34 2017

@author: albert

use naive optimization procedure
"""

import numpy as np
import pandas as pd
import scipy as sp

import datetime

import plotly.offline as offline
from plotly.graph_objs import Layout, Contour, Scatter, Heatmap

np.seterr(all="raise")


def get_history(filename="2017-06-11", end_date="", delta_t=50, length=150, col="price"):
    data = pd.read_csv("../data/" + filename, parse_dates=["Time"])
    data["price"] = data[col]
    data["time"] = np.arange(data.shape[0])
    data = data[data.Time <= end_date]
    if delta_t == 0:
        return data[-length:][["time", "price"]]
    return data[-delta_t-length: -delta_t][["time", "price"]]


def simulate_LPPLS(delta_t):
    A, B, C, tc, m, w, phi, sig = 8, -.015, .0015, 1080.01, .8, 9, 0.0, .03    
    t = np.arange(int(tc) - delta_t - 800, int(tc) - delta_t)
    LPPLS = A + \
        B * np.power(np.abs(tc - t), m) + \
        C * np.power(np.abs(tc - t), m) * np.cos(w * np.log(np.abs(tc - t)) + phi)
    
    return np.vstack([t, np.exp(LPPLS + np.random.normal(0, sig, len(t)))]).T 


class LPPLS_density():
    def __init__(self):
        # clean up default graph
        print("Finish initializing ...")

    def get_LPPLS(self, t, tc, A, B, C1, C2, m, w):
        X1 = np.power(np.abs(tc - t), m)
        X2 = np.log(np.abs(tc - t))
        X3 = np.cos(w*X2)
        X4 = np.sin(w*X2)

        return A + B*X1 + C1*X1*X3 + C2*X1*X4

    # *********************************************
    # * Jacobian and Hessian for tc
    # *********************************************
    def get_X(self, t, tc, A, B, C1, C2, m, w):
        X1 = np.power(np.abs(tc - t), m)
        X2 = np.log(np.abs(tc - t))
        X3 = np.cos(w*X2)
        X4 = np.sin(w*X2)
        X = np.array([np.ones(len(t)),
                      X1, X1*X3, X1*X4, X1*X2*( B + C1*X3 + C2*X4 ), X1*X2*(-C1*X4+C2*X3)])

        return X.T

    def get_H(self, t, tc, A, B, C1, C2, m, w):
        X1 = np.power(np.abs(tc - t), m)
        X2 = np.log(np.abs(tc - t))
        X3 = np.cos(w*X2)
        X4 = np.sin(w*X2)
        H = np.zeros([len(t), 6, 6])
        H[:, 1, 4] = H[:, 4, 1] = X1*X2
        H[:, 2, 4] = H[:, 4, 2] = X1*X2*X3
        H[:, 2, 5] = H[:, 5, 2] = -X1*X2*X4
        H[:, 3, 4] = H[:, 4, 3] = X1*X2*X4
        H[:, 3, 5] = H[:, 5, 3] = X1*X2*X3
        H[:, 4, 4] = X1*X2*X2*(B + C1*X3 + C2*X4)
        H[:, 4, 5] = H[:, 5, 4] = X1*X2*X2*(-C1*X4 + C2*X3)
        H[:, 5, 5] = -X1*X2*X2*(C1*X3 + C2*X4)

        return H

    # *********************************************
    # * Jacobian and Hessian for tc, m
    # *********************************************
    def get_X_m(self, t, tc, A, B, C1, C2, m, w):
        X1 = np.power(np.abs(tc - t), m)
        X2 = np.log(np.abs(tc - t))
        X3 = np.cos(w * X2)
        X4 = np.sin(w * X2)
        X = np.array([np.ones(len(t)), X1, X1 * X3, X1 * X4, X1 * X2 * (-C1 * X4 + C2 * X3)])

        return X.T

    def get_H_m(self, t, tc, A, B, C1, C2, m, w):
        X1 = np.power(np.abs(tc - t), m)
        X2 = np.log(np.abs(tc - t))
        X3 = np.cos(w * X2)
        X4 = np.sin(w * X2)
        H = np.zeros([len(t), 5, 5])
        H[:, 2, 4] = H[:, 4, 2] = -X1 * X2 * X4
        H[:, 3, 4] = H[:, 4, 3] = X1 * X2 * X3
        H[:, 4, 4] = -X1 * X2 * X2 * (C1 * X3 + C2 * X4)

        return H

    # *********************************************
    # * Jacobian and Hessian for tc, w
    # *********************************************
    def get_X_w(self, t, tc, A, B, C1, C2, m, w):
        X1 = np.power(np.abs(tc - t), m)
        X2 = np.log(np.abs(tc - t))
        X3 = np.cos(w * X2)
        X4 = np.sin(w * X2)
        X = np.array([np.ones(len(t)), X1, X1 * X3, X1 * X4, X1 * X2 * (B + C1 * X3 + C2 * X4)])

        return X.T

    def get_H_w(self, t, tc, A, B, C1, C2, m, w):
        X1 = np.power(np.abs(tc - t), m)
        X2 = np.log(np.abs(tc - t))
        X3 = np.cos(w * X2)
        X4 = np.sin(w * X2)
        H = np.zeros([len(t), 5, 5])
        H[:, 1, 4] = H[:, 4, 1] = X1 * X2
        H[:, 2, 4] = H[:, 4, 2] = X1 * X2 * X3
        H[:, 3, 4] = H[:, 4, 3] = X1 * X2 * X4
        H[:, 4, 4] = X1 * X2 * X2 * (B + C1 * X3 + C2 * X4)

        return H

    def get_log_lm_m(self, t, tc, Pt, mle_params, m_params, m_sse):
        jacob = self.get_X_m(t, tc, *m_params)
        jacob_mle = self.get_X_m(t, tc, *mle_params)
        hess = self.get_H_m(t, tc, *m_params)
        err = (np.log(Pt) - self.get_LPPLS(t, tc, *m_params))[:, None, None]
        hess = np.sum(err * hess, 0)
        log_Lm = 0.5 * np.linalg.slogdet(np.matmul(jacob.T, jacob) - hess)[1]
        log_Lm -= np.linalg.slogdet(np.matmul(jacob_mle.T, jacob))[1]
        log_Lm -= (len(Pt) - 7) / 2 * np.log(m_sse)
        return log_Lm

    def get_log_lm_w(self, t, tc, Pt, mle_params, w_params, w_sse):
        jacob = self.get_X_w(t, tc, *w_params)
        jacob_mle = self.get_X_w(t, tc, *mle_params)
        hess = self.get_H_w(t, tc, *w_params)
        err = (np.log(Pt) - self.get_LPPLS(t, tc, *w_params))[:, None, None]
        hess = np.sum(err * hess, 0)
        log_Lm = 0.5 * np.linalg.slogdet(np.matmul(jacob.T, jacob) - hess)[1]
        log_Lm -= np.linalg.slogdet(np.matmul(jacob_mle.T, jacob))[1]
        log_Lm -= (len(Pt) - 7) / 2 * np.log(w_sse)
        return log_Lm

    # *********************************************
    # * SSE function subordinating m, w to tc / Formula (15)
    # *********************************************
    def lppls_cost(self, params, tc, t, pt):
        a, b, c1, c2, m, w = params
        n = len(pt)
        y = np.log(pt)

        x1 = np.power(np.abs(tc - t), m)
        x2 = w*np.log(np.abs(tc - t))
        lppls = a + b*x1 + c1*x1*np.cos(x2) + c2*x1*np.sin(x2)
        sse = np.sum(np.power(y - lppls, 2))

        return sse

    def optimize_mutli(self, tc, t, pt, retry=30):
        from scipy import optimize
        counter = 0
        best_sse = np.inf
        best_params = [None] * 6
        while True:
            b0 = -np.random.standard_exponential(1)[0]
            m0 = np.random.uniform(0.1, 0.9)
            w0 = np.random.uniform(6, 13)
            try:
                res = optimize.minimize(self.lppls_cost,
                                        x0=np.array([0, b0, 1, 1, m0, w0]),
                                        bounds=[[-np.inf, np.inf],
                                                [-np.inf, np.inf],
                                                [-np.inf, np.inf],
                                                [-np.inf, np.inf],
                                                [-1, 2],
                                                [1, 20]],
                                        args=(tc, t, pt),
                                        method="L-BFGS-B",
                                        tol=1E-6)
                if res.fun < best_sse:
                    best_sse = res.fun
                    best_params = res.x
                counter += 1
            except ValueError:
                pass

            if counter > retry:
                break

        return best_sse, best_params

    # *********************************************
    # * full MLEs / Formula (14)
    # *********************************************
    class Result:
        def __init__(self, crash=None, SSE=np.inf, params=None):
            self.crash = crash
            self.SSE = SSE
            self.params = params

    def get_mle_linear(self, crash_times, t, pt):
        results = []
        best_result = self.Result()
        counter = 0
        for crash in crash_times:
            sse, params = self.optimize_mutli(crash, t, pt)
            results.append(self.Result(crash, sse, params))

            if results[-1].SSE < best_result.SSE:
                best_result = results[-1]

            counter += 1
            print("\rCalculating MLEs: {:.0f}%".format(counter / len(crash_times) * 100), end="")
        print()

        return best_result, results

    def get_density(self, timestamps, prices, ref_date):
        N = len(prices)
        # plus 0.01 to avoid singularity when tc = t
        cob_date = timestamps[-1]
        crash_times = np.arange(cob_date-10, cob_date+50) + 0.01

        print("------------------- ")
        print("Length:", N, flush=True)

        # *********************************************
        # * evaluate density value at each crash time point
        # *********************************************    
        mle, results = self.get_mle_linear(crash_times, timestamps, prices)
        print("tc:", mle.crash, " | SSE:", mle.SSE/N)

        log_Lm = []
        F2s = [ res.SSE for res in results ]
        # ref_date = datetime.date(2015, 6, 12)
        ms = [ res.params[4] for res in results ]
        ws = [ res.params[5] for res in results ]
        Ds = []
        tcs = []
        filter = []

        Xmm = self.get_X(timestamps, mle.crash, *mle.params)
        for counter, res in enumerate(results):
            # calculate likelihood for tc
            Xm = self.get_X(timestamps, res.crash, *res.params)
            Hm = self.get_H(timestamps, res.crash, *res.params)
            Err = (np.log(prices) - self.get_LPPLS(timestamps, res.crash, *res.params))[:, None, None]
            Hm = np.sum(Err*Hm, 0)
            log_Lm1 = 0.5 * np.linalg.slogdet(np.matmul(Xm.T, Xm) - Hm)[1]
            log_Lm2 = 0.5 * np.linalg.slogdet(np.matmul(Xmm.T, Xm))[1]
            log_Lm.append(log_Lm1 - log_Lm2 - (N - 8)/2*np.log(res.SSE))

            _, B, C1, C2, m, w = res.params
            Ds.append( m*abs(B) / w / np.sqrt(C1**2 + C2**2))
            tcs.append(ref_date + datetime.timedelta(days=np.floor(res.crash)-cob_date))

            # filtering using m and w
            valid = 0.1 < res.params[4] < 0.9 and 6 < res.params[5] < 13

            filter.append(valid)
            print("\rCalculating filter: {:.0f}%".format((counter + 1) / len(results) * 100), end="")
        print()

        log_Lm = np.array(log_Lm)
        Lm = np.exp(log_Lm - np.max(log_Lm))
        return [F2s, Lm, np.abs(ms), np.abs(ws), Ds, tcs, filter]


# data = pd.read_csv("../data/000001.SS.csv")
sample_sizes = np.arange(150, 500, 50)
lm_all = []
keep_all = []
density = LPPLS_density()
for length in sample_sizes:
    data = get_history("bitcoinity_data.csv", "2017-05-25", delta_t=0, length=length, col="itbit")
    F2s, Lm, ms, ws, Ds, tcs, keep = density.get_density(data.time.as_matrix(),
                                                         data.price.as_matrix(),
                                                         datetime.date(2017, 5, 25))
    lm_all.append(Lm)
    keep_all.append(keep)

keep_all = 1 - np.array(keep_all).astype(int)

# *********************************************
# * Plotting
# *********************************************
my_color = [
    [0, "rgb(255, 255, 255)"],
    [0.4, "rgb(255, 240, 240)"],
    [0.7, "rgb(255, 179, 179)"],
    [1, "rgb(255, 102, 102)"]
]


filter_color = [
    [0, "rgb(255, 255, 255)"],
    [0.999, "rgb(255, 255, 255)"],
    [1, "rgb(0, 0, 255)"]
]

offline.plot({
    "data": [
        Contour(z=lm_all, y=sample_sizes, colorscale=my_color),
        Contour(z=keep_all, y=sample_sizes, colorscale=filter_color, opacity=0.5, contours=dict(showlines=False)),
    ],
    # "layout": layout
})

# fig = tools.make_subplots( rows=4, cols=1 )
# fig.append_trace( Scatter(x=tcs, y=F2s), 1, 1 )
# fig.append_trace( Scatter(x=tcs, y=Lm, name="Modified LP"), 2, 1 )
# fig.append_trace( Scatter(x=tcs, y=Lmp, name="LP" ), 2, 1)
# fig.append_trace( Scatter(x=tcs, y=ws), 3, 1)
# fig.append_trace( Scatter(x=tcs, y=Ds), 4, 1)

# offline.plot( fig )
