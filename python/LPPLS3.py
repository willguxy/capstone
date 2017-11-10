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
from plotly.graph_objs import Layout, Contour, Scatter
from plotly import tools

np.seterr(all="raise")


def get_history(date="2017-06-11", delta_t=50):
    data = pd.read_csv("../data/" + date + ".csv", delimiter="\t")
    return np.array(data[-delta_t - 701:-(delta_t + 1)])


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
                raise ValueError( "X is singular" )
                
            SSE = np.sum(np.power(y - np.matmul(beta, X.T), 2))
            if flag:
                return SSE, beta
            return SSE
    
        # multiple starting points to mitigate local extrema
        retrys = 15
        counter = 0
        best_SSE = np.inf

        while 1:
            m = np.random.uniform(0.1, 0.9)
            w = np.random.uniform(6, 13)
            try:
                res = sp.optimize.minimize(F1, x0=[m, w], method="BFGS", tol=1E-6)                
                if res.fun < best_SSE:
                    best_SSE = res.fun
                    params = res.x                    
                counter += 1   
            except:
                pass                     

            if counter > retrys:
                break

        if flag:
            _, ABCC = F1(params, True)
            return True, best_SSE, np.hstack([ABCC, params])
        return True, best_SSE
    
    
    # *********************************************
    # * full MLEs / Formula (14)
    # *********************************************
    def get_MLEs(self, t, Pt):       
        res = sp.optimize.minimize_scalar(self.F2, 
                                          bracket = (t[-1] - 50.01, t[-1]+200.01), 
                                          args=(t, Pt),
                                          method="golden", 
                                          tol=1E-6)           
        _, ABCCmw = self.F2(res.x, t, Pt, True)
        return res.fun, res.x, ABCCmw
    
    def get_MLEs_linear(self, crash_times, t, Pt):      
        class Result:
            def __init__(self, crash=None, SSE=np.inf, params=None):
                self.crash = crash
                self.SSE = SSE
                self.params = params
            
        results = []
        best_result = Result()
        counter = 0
        for crash in crash_times:
            valid, SSE, params = self.F2( crash, t, Pt, True )
            if not valid:
                break
            
            results.append( Result( crash, SSE, params ) )
            
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
        crash_times = np.arange(cob_date-10, cob_date+60) + 0.01
        
        print("------------------- ")
        print("Length:", N)
            
        # *********************************************
        # * evaluate density value at each crash time point
        # *********************************************    
        mle, results = self.get_MLEs_linear(crash_times, timestamps, prices)
        print( mle.crash )
        print( mle.SSE )
        
        log_Lm = []
        F2s = [ res.SSE for res in results ]
        ref_date = datetime.date(2015, 6, 12)
        tcs = []
        ms = [ res.params[4] for res in results ]
        ws = [ res.params[5] for res in results ]
        Ds = []
        tcs = []
        
        Xmm = self.get_X(timestamps, mle.crash, *mle.params)
        for res in results:
            Xm = self.get_X(timestamps, res.crash, *res.params)          
            Hm = self.get_H(timestamps, res.crash, *res.params)
            Err = (np.log( prices ) - self.get_LPPLS(timestamps, res.crash, *res.params))[:, None, None]
            Hm = np.sum( Err*Hm, 0 )
            log_Lm1 = 0.5 * np.linalg.slogdet( np.matmul( Xm.T, Xm ) - Hm )[1]
            log_Lm2 = 0.5 * np.linalg.slogdet( np.matmul( Xmm.T, Xm ) )[1]
            log_Lm.append( log_Lm1 - log_Lm2 - ( N - 8 )/2*np.log( res.SSE ) )
            
            _, B, C1, C2, m, w = res.params
            Ds.append( m*abs(B) / w / np.sqrt(C1**2 + C2**2))
            tcs.append( ref_date + datetime.timedelta( days=np.floor(res.crash)-cob_date ) )
                
        log_Lm = np.array( log_Lm )
        Lm = np.exp( log_Lm - np.max( log_Lm ) )
        Lmp = np.power( F2s, -N/2 )
        return [ F2s, Lm, Lmp/np.max( Lmp ), np.abs( ms ), np.abs( ws ), Ds, tcs ]


data = pd.read_csv("../data/000001.SS.csv")
prices = data["Adj Close"]
times = np.arange( len( prices ) )
density = LPPLS_density()
F2s, Lm, Lmp, ms, ws, Ds, tcs = density.get_density(times, prices)
print(tcs)
print(F2s)
print(Lm)

# *********************************************
# * Plotting
# *********************************************

fig = tools.make_subplots( rows=4, cols=1 )
fig.append_trace( Scatter(x=tcs, y=F2s), 1, 1 )
fig.append_trace( Scatter(x=tcs, y=Lm, name="Modified LP"), 2, 1 )
fig.append_trace( Scatter(x=tcs, y=Lmp, name="LP" ), 2, 1)
fig.append_trace( Scatter(x=tcs, y=ws), 3, 1)
fig.append_trace( Scatter(x=tcs, y=Ds), 4, 1)

offline.plot( fig )
