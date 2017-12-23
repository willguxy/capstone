"""
Density class and result class defintion
"""

import numpy as np
import scipy as sp
import datetime


class Result:
    def __init__(self, crash=None, sse=np.inf, params=None):
        self.crash = crash
        self.sse = sse
        self.params = params


class LPPLSDensity:
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
                      X1, X1*X3, X1*X4,
                      X1*X2*(B + C1*X3 + C2*X4),
                      X1*X2*(-C1*X4+C2*X3)])

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
    def get_f2(self, tc, m0, w0, t, Pt, flag=False):
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
        retries = 20
        counter = 0
        succeeds = 0
        best_sse = np.inf

        from scipy import optimize

        while 1:
            m = m0 if m0 else np.random.uniform(0.1, 0.9)
            w = w0 if w0 else np.random.uniform(6, 13)
            try:
                if m0 or w0:
                    res = optimize.minimize(F1, x0=[w if m0 else m], method="Nelder-Mead", tol=1E-6)
                else:
                    res = optimize.minimize(F1, x0=[m, w], method="Nelder-Mead", tol=1E-6)
                if res.fun < best_sse:
                    best_sse = res.fun
                    if m0 or w0:
                        res.x = np.hstack([m0, res.x]) if m0 else np.hstack([res.x, w0])
                    params = res.x
                succeeds += 1
                counter = 0
            except ValueError:
                pass

            counter += 1
            if counter > retries:
                if flag:
                    return False, None, None
                return False, None

            if succeeds > retries:
                break

        if flag:
            _, ABCC = F1(params, True)
            return True, best_sse, np.hstack([ABCC, params])
        return True, best_sse

    # *********************************************
    # * full MLEs / Formula (14)
    # *********************************************
    def get_mle_linear(self, crash_times, t, Pt):
        results = []
        best_result = Result()
        counter = 0
        for crash in crash_times:
            succeed, SSE, params = self.get_f2(crash, None, None, t, Pt, True)
            results.append(Result(crash, SSE, params))

            if results[-1].sse < best_result.sse:
                best_result = results[-1]

            counter += 1
            print("\rCalculating MLEs: {:.0f}%".format(counter / len(crash_times) * 100), end="")
        print()

        return best_result, results

    def get_density(self, timestamps, prices, ref_date, delta_t):
        N = len(prices)
        # plus 0.01 to avoid singularity when tc = t
        cob_date = timestamps[-1]
        crash_times = np.arange(cob_date-10, cob_date+150) + 0.01

        print("------------------- ")
        print("{} Delta:{} Length:{}".format(ref_date, delta_t, N), flush=True)

        # *********************************************
        # * evaluate density value at each crash time point
        # *********************************************
        mle, results = self.get_mle_linear(crash_times, timestamps, prices)
        print("tc:", mle.crash, " | SSE:", mle.sse / N)

        log_Lm = []
        F2s = [res.sse for res in results]
        # ref_date = datetime.date(2015, 6, 12)
        ms = [res.params[4] for res in results]
        ws = [res.params[5] for res in results]
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
            log_Lm.append(log_Lm1 - log_Lm2 - (N - 8)/2*np.log(res.sse))

            _, B, C1, C2, m, w = res.params
            Ds.append(m*abs(B) / w / np.sqrt(C1**2 + C2**2))
            tcs.append(ref_date + datetime.timedelta(days=np.floor(res.crash) - cob_date - delta_t))

            # calculate likelihood for m, w
            valid = True

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
                succeed, m_sse, m_params = self.get_f2(res.crash, m_test, None, timestamps, prices, True)
                if succeed:
                    lm_m = self.get_log_lm_m(timestamps, res.crash, prices, res.params, m_params, m_sse) \
                                  - self.get_log_lm_m(timestamps, res.crash, prices, res.params, res.params, res.sse)
                    valid &= lm_m > np.log(0.05)
                else:
                    valid = False

            if valid and w_test:
                succeed, w_sse, w_params = self.get_f2(res.crash, None, w_test, timestamps, prices, True)
                if succeed:
                    lm_w = self.get_log_lm_w(timestamps, res.crash, prices, res.params, w_params, w_sse) \
                                  - self.get_log_lm_w(timestamps, res.crash, prices, res.params, res.params, res.sse)
                    valid &= lm_w > np.log(0.05)
                else:
                    valid = False

            filter.append(valid)
            print("\rCalculating filter: {:.0f}%".format((counter + 1) / len(results) * 100), end="")
        print()

        log_Lm = np.array(log_Lm)
        Lm = np.exp(log_Lm - np.max(log_Lm))
        return [F2s, Lm, np.abs(ms), np.abs(ws), Ds, tcs, filter]