# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 23:05:34 2017

@author: albert

Use synthetic data to unit test the code
"""

import numpy as np
import pandas as pd
import scipy as sp

import datetime

import plotly.offline as offline
from plotly.graph_objs import Layout, Contour, Scatter, Heatmap

np.seterr(all="raise")


def get_history(filename="2017-06-11", delta_t=50, length=150, col="price"):
    data = pd.read_csv("../data/" + filename)
    data["price"] = data[col]
    data["time"] = np.arange(data.shape[0])
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
    def F2(self, tc, m0, w0, t, Pt, flag=False):
        N = len(Pt)
        y = np.log(Pt)

        # subordinating A, B, C1, C2 to m, w
        def F1(params, flag=False):
            if m0:
                _m, _w = m0, params[0]
            elif w0:
                _m, _w = params[0], w0
            else:
                _m, _w = params

            tmp = np.power(np.abs(tc - t), _m)

            X = [np.ones(N),
                 tmp,
                 tmp * np.cos(_w * np.log(np.abs(tc - t))),
                 tmp * np.sin(_w * np.log(np.abs(tc - t)))]

            X = np.vstack(X).T
            try:
                beta = sp.matmul(sp.linalg.inv(sp.matmul(X.T, X)), sp.matmul(X.T, y))
            except:
                # print(_m, _w)
                raise ValueError("X is singular")

            SSE = np.sum(np.power(y - np.matmul(beta, X.T), 2))
            if flag:
                return SSE, beta
            return SSE

        # multiple starting points to mitigate local extrema
        retrys = 15
        counter = 0
        best_SSE = np.inf

        from scipy import optimize

        while 1:
            m = m0 if m0 else np.random.uniform(0.1, 0.9)
            w = w0 if w0 else np.random.uniform(6, 13)
            try:
                if m0 or w0:
                    res = optimize.minimize(F1, x0=[w if m0 else m], method="Nelder-Mead", tol=1E-6)
                else:
                    res = optimize.minimize(F1, x0=[m, w], method="Nelder-Mead", tol=1E-6)
                if res.fun < best_SSE:
                    best_SSE = res.fun
                    if m0 or w0:
                        res.x = np.hstack([m0, res.x]) if m0 else np.hstack([res.x, w0])
                    params = res.x
                counter += 1
            except ValueError:
                pass

            if counter > retrys:
                break

        if flag:
            _, ABCC = F1(params, True)
            return best_SSE, np.hstack([ABCC, params])
        return best_SSE

    # *********************************************
    # * full MLEs / Formula (14)
    # *********************************************
    class Result:
        def __init__(self, crash=None, SSE=np.inf, params=None):
            self.crash = crash
            self.SSE = SSE
            self.params = params

    def get_MLEs_linear(self, crash_times, t, Pt):
        results = []
        best_result = self.Result()
        counter = 0
        for crash in crash_times:
            SSE, params = self.F2(crash, None, None, t, Pt, True)
            results.append(self.Result(crash,SSE,params))

            if results[-1].SSE < best_result.SSE:
                best_result = results[-1]

            counter += 1
            print("\rCalculating MLEs: {:.0f}%".format(counter / len(crash_times) * 100), end="")
        print()

        return best_result, results

    def get_density(self, timestamps, prices):
        N = len(prices)
        # plus 0.01 to avoid singularity when tc = t
        cob_date = timestamps[-1]
        crash_times = np.arange(cob_date-10, cob_date+50) + 0.01

        print("------------------- ")
        print("Length:", N, flush=True)

        # *********************************************
        # * evaluate density value at each crash time point
        # *********************************************    
        mle, results = self.get_MLEs_linear(crash_times, timestamps, prices)
        print("tc:", mle.crash, " | SSE:", mle.SSE/N)

        log_Lm = []
        F2s = [ res.SSE for res in results ]
        ref_date = datetime.date(2015, 6, 12)
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

            # calculate likelihood for m, w
            valid = True
            if res.params[5] < 3:
                valid = False

            if res.params[4] > 0.9:
                m_test = 0.9
            elif res.params[4] < 0.1:
                m_test = 0.1
            else:
                m_test = None

            if res.params[5] > 13:
                w_test = 13
            elif res.params[5] < 6:
                w_test = 6
            else:
                w_test = None

            if valid and m_test:
                m_sse, m_params = self.F2(res.crash, m_test, None, timestamps, prices, True)
                lm_m = np.exp(self.get_log_lm_m(timestamps, res.crash, prices, res.params, m_params, m_sse)
                              - self.get_log_lm_m(timestamps, res.crash, prices, res.params, res.params, res.SSE))
                valid &= lm_m > 0.05

            if valid and w_test:
                w_sse, w_params = self.F2(res.crash, None, w_test, timestamps, prices, True)
                lm_w = np.exp(self.get_log_lm_w(timestamps, res.crash, prices, res.params, w_params, w_sse)
                              - self.get_log_lm_w(timestamps, res.crash, prices, res.params, res.params, res.SSE))
                valid &= lm_w > 0.05

            filter.append(valid)
            print("\rCalculating filter: {:.0f}%".format((counter + 1)/ len(results) * 100), end="")
        print()

        log_Lm = np.array(log_Lm)
        Lm = np.exp(log_Lm - np.max(log_Lm))
        return [F2s, Lm, np.abs(ms), np.abs(ws), Ds, tcs, filter]


# data = pd.read_csv("../data/000001.SS.csv")
sample_sizes = np.arange(100, 750, 50)
lm_all = []
keep_all = []
density = LPPLS_density()
for length in sample_sizes:
    data = get_history("000001.SS.csv", delta_t=0, length=length, col="Adj Close")
    F2s, Lm, ms, ws, Ds, tcs, keep = density.get_density(data.time.as_matrix(), data.price.as_matrix())
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
