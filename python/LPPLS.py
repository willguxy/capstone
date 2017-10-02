# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 23:05:34 2017

@author: alber
"""

import tensorflow as tf
import numpy as np
import pandas as pd


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

# set up variables in tensorflow's graph
A = tf.Variable(8.0, name="A")
B = tf.Variable(-0.015, name="B")
C1 = tf.Variable(0.0010, name="C1")
C2 = tf.Variable(0.0005, name="C2")

# we will supply tc values later on, hence use placeholder instead of variable
tc = tf.placeholder(dtype=tf.float32, shape=1, name="tc")
m = tf.Variable(0.8, name="m")
w = tf.Variable(9.0, name="w")

# we will suuply data to these two variables, hence use placeholder
t = tf.placeholder(dtype=tf.float32, shape=None, name="t")
Pt = tf.placeholder(dtype=tf.float32, shape=None, name="Pt")

# the original LPPLS formula
LPPLS = A + \
        B * tf.pow(tf.abs(tc - t), m) + \
        C1 * tf.pow(tf.abs(tc - t), m) * tf.cos(w * tf.log(tf.abs(tc - t))) + \
        C2 * tf.pow(tf.abs(tc - t), m) * tf.sin(w * tf.log(tf.abs(tc - t)))
    
X = tf.gradients(LPPLS, [A, B, C1, C2, m, w])  # formula (36)
H = (tf.log(Pt) - LPPLS) * compute_hessian(LPPLS, [A, B, C1, C2, m, w])  # formula (37)
SSE = tf.reduce_mean( tf.pow(tf.log(Pt) - LPPLS, 2) )  # formula (9)

# run optimization to get MLEs
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
training_op = optimizer.minimize(SSE)

# initialize all variables defined above
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    # read and prepare data
    data = pd.read_csv("../data/SPX.csv", infer_datetime_format=True)
    Pt0 = data[data.Date < "2007-11-01"]["Adj Close"]  # 2007-11-01 was the peak
    N = len(Pt0)
    t0 = np.arange(N)  # timestamp from 0 to length - 1    
    tc0 = np.arange(N-50, N+100) + 0.1  # plus 0.1 to avoid singularity when tc = t
    
    sess.run(init_op)  # run initialization step
    
    log_Lm = []  # to hold log-likelihood
    for tmp_tc in tc0:
        # get MLEs
        curr, prev = -1, 0
        while abs(curr - prev) > 1E-6:
            prev = curr
            _, curr = sess.run([training_op, SSE], 
                               feed_dict={t: t0, Pt: Pt0, tc: [tmp_tc]})
            
        Ah, Bh, C1h, C2h, mh, wh = sess.run([A, B, C1, C2, m, w])
        s_tc = curr
        
        # calculate gradient matrix X
        X_mat = []
        for tmp_t in t0:
            row = sess.run( X, feed_dict={t: [tmp_t], tc: [tmp_tc]} )
            row = np.hstack( row )
            X_mat.append( row )
        X_mat = np.vstack( X_mat )
        
        # calculate Hessian matrix H
        H_mat = np.zeros([6, 6])
        for tmp_t, tmp_p in zip( t0, Pt0 ):
            row = sess.run( H, feed_dict={t: [tmp_t], Pt: [tmp_p], tc: [tmp_tc]} )
            H_mat += np.squeeze( row )  # squeeze to remove redundant dimensions
        
        # calculate likelihold Lm
        tmp = 0.5 * np.log(np.abs(np.linalg.det(np.dot(X_mat.T, X_mat) - H_mat))) - \
                    np.log(np.linalg.det(np.dot(X_mat.T, X_mat))) + \
                    -.5*(N - 8) * np.log(s_tc)
        log_Lm.append(tmp)
        print( "tc:", tmp_tc, "   log-likelihood:", tmp )
        
    log_Lm = np.array(log_Lm)
    Lm = np.exp(log_Lm - np.max(log_Lm))  # normalize the log-likelihood and transform back to likelihood
    pd.DataFrame({"tc": tc0, "Lm": Lm}).to_csv("Lm.csv", index=False)  # output