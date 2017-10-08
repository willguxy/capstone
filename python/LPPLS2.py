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
from plotly.graph_objs import Scatter

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
        # represent A, B, C1, C2, m, w respectively
        params = tf.Variable([8.0, -.015, .001, .0005, .8, 9.0])
        A, B, C1, C2, m, w = tf.split(params, 6)
        
        # damping parameter
        self.D = m * tf.abs(B) / w / tf.sqrt(tf.pow(C1, 2) + tf.pow(C2, 2))  
    
        # we will suuply data to these two variables, hence use placeholder
        self.tc = tf.placeholder(dtype=tf.float32, shape=1, name="tc")
        self.t = tf.placeholder(dtype=tf.float32, shape=None, name="t")
        self.Pt = tf.placeholder(dtype=tf.float32, shape=None, name="Pt")
    
        # the LPPLS formula
        def get_LPPLS(a, b, c1, c2, _m, _w, _tc, _t):
            p1 = b * tf.pow(tf.abs(_tc - _t), _m)
            p2 = c1 * tf.pow(tf.abs(_tc - _t), _m) * tf.cos(_w * tf.log(tf.abs(_tc - _t)))
            p3 = c2 * tf.pow(tf.abs(_tc - _t), _m) * tf.sin(_w * tf.log(tf.abs(_tc - _t)))
            return a + p1 + p2 + p3
        
        LPPLS = get_LPPLS(A, B, C1, C2, m, w, self.tc, self.t)
                
        self.X = tf.gradients(LPPLS, params)  # formula (36) 
        self.H = (tf.log(self.Pt) - LPPLS) * tf.hessians(LPPLS, params)  # formula (37)
    
        self.SSE = tf.reduce_mean( tf.pow(tf.log(self.Pt) - LPPLS, 2) )  # formula (9)
    
    
        # *********************************************
        # * use optimization to get MLEs
        # *********************************************
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.training_op = optimizer.minimize(self.SSE)
    
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
            beta = sp.matmul(sp.linalg.inv(sp.matmul(X.T, X)), sp.matmul(X.T, y))
            SSE = np.sum(np.power(y - np.matmul(beta, X.T), 2))
            if flag:
                return SSE, beta
            return SSE
    
        # multiple starting points to mitigate local extrema
        ms = [.1, .5, .9]
        ws = [4, 7, 10]
        best_SSE = np.inf
        for m in ms:
            for w in ws:
                res = sp.optimize.minimize(F1, x0=[m, w], method="Nelder-Mead", tol=1E-6)
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


    def get_density(self, prices):
        with tf.Session() as sess:
            sess.run(self.init_op)  # run initialization step
            N = len(prices)
            print("------------------- ")
            print("Length:", N)
    
            timestamps = np.arange(N) + 1
            crash_times = np.arange(N-50, N+150) + 0.01  # plus 0.01 to avoid singularity when tc = t
            
            # *********************************************
            # * evaluate conditional MLEs on specific tc value
            # *********************************************
            SSE_min = np.inf
            X_mat_holder, H_mat_holder, s_tc_holder = [], [], []
            for i, crash_time in enumerate(crash_times):
                # get MLEs
                curr, prev = -1, 0
                while abs(curr - prev) > 1E-6:
                    prev = curr
                    _, curr = sess.run([self.training_op, self.SSE], 
                                       feed_dict={self.t: timestamps, 
                                                  self.Pt: prices, 
                                                  self.tc: [crash_time]})                
                s_tc_holder.append(curr)
                
                # calculate gradient matrix X
                X_mat = []
                H_mat = np.zeros([6, 6])
                for _t, _Pt in zip(timestamps, prices):
                    tmp1, tmp2 = sess.run([self.X, self.H], 
                                          feed_dict={self.t: _t,
                                                     self.Pt: _Pt,
                                                     self.tc: [crash_time]}
                                          )
    
                    X_mat.append(tmp1)
                    H_mat += tmp2[0]
                    
                X_mat = np.squeeze(np.array(X_mat))
                X_mat_holder.append(X_mat)
                H_mat_holder.append(H_mat)
                
                # find the full MLEs
#                if curr < SSE_min:
                if crash_time == N + 50.01:
                    SSE_min = curr
                    X_full = X_mat
                    tc_full = crash_time
                    D_full = sess.run(self.D)[0]
                    
                print("\r{:.0f}%".format(i / len(crash_times) * 100), end="")
             
            # *********************************************
            # * calculate likelihold Lm
            # ********************************************* 
            print("\nD:{:.2f}  tc hat:{:.2f}".format(D_full, tc_full-N))
            log_Lm = []
            for X_mat, H_mat, s_tc in zip(X_mat_holder, H_mat_holder, s_tc_holder):
                num = 0.5 * np.linalg.slogdet(np.matmul(X_mat.T, X_mat) - H_mat)[1]
                den = np.linalg.slogdet(np.matmul(X_full.T, X_mat))[1]
                log_Lm.append(num - den -.5*(N - 8) * np.log(s_tc))
                
            log_Lm = np.squeeze(np.array(log_Lm))
            # normalize the log-likelihood and floor the value to avoid underflow
            return np.exp(np.maximum(-50, log_Lm - np.max(log_Lm)))


# use synthetic data with different data lengths
raw = simulate_LPPLS(50)
density = LPPLS_density()
tmp = density.get_MLEs(raw[:, 0], raw[:, 1])
print(tmp)
#data = []
#for i in [100, 200, 250, 300, 400, 500]:
#    dat = density.get_density(raw[-i:])
#    data.append( Scatter(y=dat, name=str(i)) )
#offline.plot(data)

#Lm = np.array(Lm).T
#np.savetxt("output.csv", Lm, delimiter=",")