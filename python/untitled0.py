# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 14:51:06 2017

@author: alber
"""
import numpy as np
import pandas as pd
import plotly.offline as offline
import plotly.plotly as py
from plotly.graph_objs import Contour, Layout, Figure, Scatter


def get_history(date="2017-06-11", delta_t=50):
    data = pd.read_csv("../data/" + date + ".csv", delimiter="\t")
    return np.array(data[-delta_t - 700:-delta_t])
    
    
#dat = get_history(date="2017-06-11", delta_t=1)
#
#print(dat.shape)
#    
#offline.plot([Scatter(y=dat[:, 1])])