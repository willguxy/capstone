# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 23:05:34 2017

@author: albert

Use synthetic data to unit test the code
"""

import tensorflow as tf
import numpy as np
import scipy as sp
import plotly.offline as offline
from plotly.graph_objs import Layout, Contour

np.seterr(all="raise")


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
        tf.reset_default_graph()
        
        # *********************************************
        # * set up tensorflow computation graph
        # *********************************************
        
        # set up variables in tensorflow's graph        
        self.params = tf.placeholder(tf.float32, shape=6)  # A, B, C1, C2, m, w respectively
        self.tc = tf.placeholder(dtype=tf.float32, shape=1, name="tc")
        self.t = tf.placeholder(dtype=tf.float32, shape=None, name="t")
        self.Pt = tf.placeholder(dtype=tf.float32, shape=None, name="Pt")
        A, B, C1, C2, m, w = tf.split(self.params, 6)
    
        # the LPPLS formula
        def get_LPPLS(a, b, c1, c2, _m, _w, _tc, _t):
            p1 = b * tf.pow(tf.abs(_tc - _t), _m)
            p2 = c1 * tf.pow(tf.abs(_tc - _t), _m) * tf.cos(_w * tf.log(tf.abs(_tc - _t)))
            p3 = c2 * tf.pow(tf.abs(_tc - _t), _m) * tf.sin(_w * tf.log(tf.abs(_tc - _t)))
            return a + p1 + p2 + p3
        
        LPPLS = get_LPPLS(A, B, C1, C2, m, w, self.tc, self.t)
                
        self.X = tf.gradients(LPPLS, self.params)  # formula (36) 
        self.H = (tf.log(self.Pt) - LPPLS) * tf.hessians(LPPLS, self.params)  # formula (37)
    
        # initialize all variables defined above
        self.init_op = tf.global_variables_initializer()
        print("Finish initializing ...")
        
        
    # *********************************************
    # * SSE function subordinating m, w to tc / Formula (15)
    # *********************************************
    def F2(self, tc, t ,Pt, flag=False):
        N = len(Pt)
        y = np.log(Pt)
        
        # subordinating A, B, C1, C2 to m, w
        def F1(params, flag=False):
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
                print(X)
                exit(1)
                
            SSE = np.mean(np.power(y - np.matmul(beta, X.T), 2))
            if flag:
                return SSE, beta
            return SSE
    
        # multiple starting points to mitigate local extrema
        ms = [.15, .5, .85]
        ws = [7.0, 9.5, 12.0]
        best_SSE = np.inf
        for m in ms:
            for w in ws:
                res = sp.optimize.minimize(F1, x0=[m, w], 
                                           bounds=((0.05, 0.95), (5.5, 13.5)),
                                           method="L-BFGS-B", 
                                           tol=1E-6)
                if res.fun < best_SSE:
                    best_SSE = res.fun
                    params = res.x
        
        if flag:
            _, ABCC = F1(params, True)
            return best_SSE, np.hstack([ABCC, params])
        return best_SSE
    
    
    # *********************************************
    # * full MLEs / Formula (14)
    # *********************************************
    def get_MLEs(self, t, Pt):       
        res = sp.optimize.minimize_scalar(self.F2, 
                                          [t[-1] - 50.01, t[-1]+200.01], 
                                          args=(t, Pt),
                                          method="Golden", 
                                          tol=1E-6)           
        _, ABCCmw = self.F2(res.x, t, Pt, True)
        return res.fun, np.hstack([ABCCmw, res.x])


    def get_density(self, data):
        with tf.Session() as sess:
            sess.run(self.init_op)  # run initialization step
            
            timestamps, prices = data[:, 0], data[:, 1]            
            N = len(prices)
            # plus 0.01 to avoid singularity when tc = t
            crash_times = np.arange(timestamps[-1] - 50, timestamps[-1] + 150) + 0.01
            
            print("------------------- ")
            print("Length:", N)
    
            # *********************************************
            # * evaluate full MLEs
            # *********************************************
            s_tc_full, params = self.get_MLEs(timestamps, prices)
            _, B, C1, C2, m, w, tc_full = params
            D_full = m * np.abs(B) / w / np.sqrt(C1**2 + C2**2)
            
            # calculate full MLE gradients
            X_full = []
            for _t, _Pt in zip(timestamps, prices):
                X_full.append( sess.run(self.X, feed_dict={self.t: _t,
                                                           self.Pt: _Pt,
                                                           self.tc: params[[-1]],
                                                           self.params: params[:-1]}) )
            X_full = np.squeeze(np.array(X_full))
            print("A:{:.2f} | B:{:.2f} | C1:{:.4f} | C2:{:.4f} | m:{:.2f} | w:{:.2f} | tc:{:.1f}".format(*params))
            print("D:{:.2f}".format(D_full, tc_full))
            
            # *********************************************
            # * evaluate density value at each crash time point
            # *********************************************
            log_Lm = []            
            for i, crash_time in enumerate(crash_times):
                s_tc, params =self.F2(crash_time, timestamps, prices, True)                
                
                # calculate gradient matrix X
                X_mat = []
                H_mat = np.zeros([6, 6])
                for _t, _Pt in zip(timestamps, prices):
                    tmp1, tmp2 = sess.run([self.X, self.H], 
                                          feed_dict={self.t: _t,
                                                     self.Pt: _Pt,
                                                     self.tc: [crash_time],
                                                     self.params: params}
                                          )
    
                    X_mat.append(tmp1)
                    H_mat += tmp2[0]
                    
                X_mat = np.squeeze(np.array(X_mat))
                num = 0.5 * np.linalg.slogdet(np.matmul(X_mat.T, X_mat) - H_mat)[1]
                den = np.linalg.slogdet(np.matmul(X_full.T, X_mat))[1]
                log_Lm.append(num - den -.5*(N - 8) * np.log(s_tc))
                    
                print("\r{:.0f}%".format(i / len(crash_times) * 100), end="")             
                
        print()
        log_Lm = np.squeeze(np.array(log_Lm))
        # normalize the log-likelihood and floor the value to avoid underflow
        return np.exp(np.maximum(-50, log_Lm - np.max(log_Lm)))


# use synthetic data with different data lengths
delta_t = 0   
n_samples = [75, 100, 150, 200, 300, 400, 500, 600, 700]

raw = simulate_LPPLS(delta_t)
density = LPPLS_density()

Lm = []
for i in n_samples:
    Lm.append( density.get_density(raw[-i:]) )
Lm = np.array(Lm)

# *********************************************
# * Plotting
# *********************************************
my_color = [
        [0, "rgb(255, 255, 255)"],
        [0.4, "rgb(255, 240, 240)"],
        [0.7, "rgb(255, 179, 179)"],
        [1, "rgb(255, 102, 102)"]
        ]

# vertical lines
layout = Layout(shapes=[
    {"x0": 50, "y0": n_samples[0], "x1": 50, "y1": n_samples[-1], "line": {"color": "#FF0000"}},
    {"x0": delta_t+50, "y0": n_samples[0], "x1": delta_t+50, "y1": n_samples[-1]}
])

offline.plot({
    "data": [Contour(z=Lm, y=n_samples, colorscale=my_color)],
    "layout": layout
})

np.savetxt("output.csv", Lm.T, delimiter=",")