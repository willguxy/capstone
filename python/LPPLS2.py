# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 23:05:34 2017

@author: albert

Use synthetic data to unit test the code
"""

import tensorflow as tf
import numpy as np
import plotly.offline as offline
from plotly.graph_objs import Data, Contour, Scatter

np.seterr(all="raise")


def simulate_LPPLS(delta_t):
    A, B, C, tc, m, w, phi, sig = 8, -.015, .0015, 1080.01, .8, 9, 0.3, .03    
    t = np.arange(int(tc) - delta_t - 800, int(tc) - delta_t)
    LPPLS = A + \
        B * np.power(np.abs(tc - t), m) + \
        C * np.power(np.abs(tc - t), m) * np.cos(w * np.log(np.abs(tc - t)) + phi)
    
    return np.exp(LPPLS + np.random.normal(0, sig, len(t)))


def compute_hessian(fn, vars):
    """
    for calculating the Hessian matrix. tensorflow's Hessians is not that convenient
    @fn: function to be taken hessian (LPPLS in this case)
    @vars: variables taken hessian with respect to
    """
    mat = []
    for v1 in vars:
        holder = []
        v1_grad = tf.gradients(fn, v1)[0]  # taking 1st gradient
        for v2 in vars:
            hess = tf.gradients(v1_grad, v2)[0]  # taking 2nd gradient which is hessian            
            holder.append(hess if hess is not None else 0.0)  # tensorflow returns None when there is no gradient
        holder = tf.stack(holder)  # assemble values into vector
        mat.append(holder)  # assemble vectors into hessian matrix
    mat = tf.stack(mat)
    return mat


#def compute_hsesian(fn, params):
    


def get_LPPLS_density(prices):
    """
    Args:
        prices (array of double): price series to be exmained. Should at least
                                    contains 75 data points
        
    Returns:
        Density of each time point from -50 of last point to +100 of last point
    """
    
    # clean up default graph
    tf.reset_default_graph()
    N = len(prices)
    
    # *********************************************
    # * set up tensorflow computation graph
    # *********************************************
    
    # set up variables in tensorflow's graph
    # represent A, B, C1, C2, m, w respectively
    params = tf.Variable([8.0, -.015, .001, .0005, .8, 9.0])
    A, B, C1, C2, m, w = tf.split(params, 6)
    D = m * tf.abs(B) / w / tf.sqrt(tf.pow(C1, 2) + tf.pow(C2, 2))  # damping parameter

    # we will suuply data to these two variables, hence use placeholder
    tc = tf.placeholder(dtype=tf.float32, shape=1, name="tc")
    t = tf.placeholder(dtype=tf.float32, shape=None, name="t")
    Pt = tf.placeholder(dtype=tf.float32, shape=None, name="Pt")

    # the LPPLS formula
    def get_LPPLS(a, b, c1, c2, _m, _w, _tc, _t):
        p1 = b * tf.pow(tf.abs(_tc - _t), _m)
        p2 = c1 * tf.pow(tf.abs(_tc - _t), _m) * tf.cos(_w * tf.log(tf.abs(_tc - _t)))
        p3 = c2 * tf.pow(tf.abs(_tc - _t), _m) * tf.sin(_w * tf.log(tf.abs(_tc - _t)))
        return a + p1 + p2 + p3
    
    LPPLS = get_LPPLS(A, B, C1, C2, m, w, tc, t)
            
    X = tf.gradients(LPPLS, params)  # formula (36) 
    H = (tf.log(Pt) - LPPLS) * tf.hessians(LPPLS, params)  # formula (37)

    SSE = tf.reduce_mean( tf.pow(tf.log(Pt) - LPPLS, 2) )  # formula (9)


    # *********************************************
    # * use optimization to get MLEs
    # *********************************************
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    training_op = optimizer.minimize(SSE)

    # initialize all variables defined above
    init_op = tf.global_variables_initializer()


    # *********************************************
    # * Start computing the density
    # *********************************************
    with tf.Session() as sess:
        sess.run(init_op)  # run initialization step
        print("start")
        print("Length of price series: ", N)

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
                _, curr = sess.run([training_op, SSE], 
                                   feed_dict={t: timestamps, 
                                              Pt: prices, 
                                              tc: [crash_time]})                
            s_tc_holder.append(curr)
            
            # calculate gradient matrix X
            X_mat = []
            H_mat = np.zeros([6, 6])
            for _t, _Pt in zip(timestamps, prices):
                X_mat.append(sess.run( X, feed_dict={t: _t, tc: [crash_time]}))
                H_mat += sess.run( H, feed_dict={t: _t, Pt: _Pt, tc: [crash_time]})[0]
                
            X_mat = np.squeeze(np.array(X_mat))
            X_mat_holder.append(X_mat)
            H_mat_holder.append(H_mat)
            
            # find the full MLEs
            if curr < SSE_min:
                SSE_min = curr
                X_full = X_mat
                tc_full = crash_time
                D_full = sess.run(D)[0]
                
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
raw = simulate_LPPLS(50)[-400:]
dat = get_LPPLS_density(raw)
data = [Scatter(y=dat)]
offline.plot(data)

#Lm = np.array(Lm).T
#np.savetxt("output.csv", Lm, delimiter=",")