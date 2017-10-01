# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 23:05:34 2017

@author: alber
"""

import tensorflow as tf
import numpy as np

def compute_hessian(fn, vars):
    mat = []
    for v1 in vars:
        holder = []
        v1_grad = tf.gradients(fn, v1)[0]
        for v2 in vars:
            hess = tf.gradients(v1_grad, v2)[0]
            # tensorflow returns None when there is no gradient
            holder.append(hess if hess is not None else tf.zeros([1], dtype=tf.float32)) 
        holder = tf.stack(holder)
        mat.append(holder)
    mat = tf.stack(mat)
    return mat

A = tf.placeholder(dtype=tf.float32, shape=1, name="A")
B = tf.placeholder(dtype=tf.float32, shape=1, name="B")
C1 = tf.placeholder(dtype=tf.float32, shape=1, name="C1")
C2 = tf.placeholder(dtype=tf.float32, shape=1, name="C2")

tc = tf.placeholder(dtype=tf.float32, shape=1, name="tc")
t = tf.placeholder(dtype=tf.float32, shape=1, name="t")
Pt = tf.placeholder(dtype=tf.float32, shape=1, name="Pt")

m = tf.placeholder(dtype=tf.float32, shape=1)
w = tf.placeholder(dtype=tf.float32, shape=1)

LPPLS = A + \
        B * tf.pow(tf.abs(tc - t), m) + \
        C1 * tf.pow(tf.abs(tc - t), m) * tf.cos(w * tf.log(tf.abs(tc - t))) + \
        C2 * tf.pow(tf.abs(tc - t), m) * tf.sin(w * tf.log(tf.abs(tc - t)))
        
X = tf.gradients(LPPLS, [A, B, C1, C2, m, w])
H = (tf.log(Pt) - LPPLS) * compute_hessian(LPPLS, [A, B, C1, C2, m, w])

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    A0, B0, C10, C20 = [8], [-0.015], [0.0015], [0.0001]
    m0, w0 = [0.8], [9]
    
    tc0 = [100]
    
    Pt0 = [100, 102, 101, 103]
    t0 = [1, 2, 3, 4]
    
    
    sess.run(init_op)
    
    # calculate gradient matrix X
    X_mat = []
    for tmp_t in t0:
        row = sess.run( X, feed_dict={A: A0, B: B0, C1: C10, C2: C20,
                                   m: m0, w: w0, t: [tmp_t], tc: tc0} )
        row = np.hstack( row )
        X_mat.append( row )
    X_mat = np.vstack( X_mat )
    
    # calculate Hessian matrix H
    H_mat = np.zeros([6, 6])
    for tmp_t, tmp_p in zip( t0, Pt0 ):
        row = sess.run( H, feed_dict={A: A0, B: B0, C1: C10, C2: C20,
                                      m: m0, w: w0, tc: tc0,
                                      t: [tmp_t], Pt: [tmp_p]} )
        H_mat += np.squeeze( row )
    
    print( H_mat )
    
#    A: A0, B: B0, C1: C10, C2: C20, m: m0, w: w0, Pt: Pt0, t: tmp_t, tc: tc0
    