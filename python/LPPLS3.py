# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 23:05:34 2017

@author: albert

Use synthetic data to unit test the code
"""

from utils import get_history
from LPPLSDensity import LPPLSDensity

import numpy as np

import datetime
import shutil

import plotly.offline as offline
from plotly.graph_objs import Layout, Contour

np.seterr(all="raise")


# data = pd.read_csv("../data/000001.SS.csv")
sample_sizes = np.arange(100, 300, 25)
cob_date = "2017-05-25"
delta_t = 50
lm_all = []
keep_all = []
density = LPPLSDensity()
for length in sample_sizes:
    data = get_history("bitcoinity_data.csv", cob_date, delta_t=0, length=length, col="kraken")
    F2s, Lm, ms, ws, Ds, tcs, keep = density.get_density(data.time.as_matrix(),
                                                         data.price.as_matrix(),
                                                         datetime.date(2017, 5, 25), delta_t)
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
        Contour(z=lm_all, y=sample_sizes, x=tcs, colorscale=my_color),
        Contour(z=keep_all, y=sample_sizes, x=tcs, colorscale=filter_color, opacity=0.5, contours=dict(showlines=False)),
    ],
    # "layout": layout
})

shutil.move("temp-plot.html", "plot-" + cob_date + "-" + str(delta_t))


# fig = tools.make_subplots( rows=4, cols=1 )
# fig.append_trace( Scatter(x=tcs, y=F2s), 1, 1 )
# fig.append_trace( Scatter(x=tcs, y=Lm, name="Modified LP"), 2, 1 )
# fig.append_trace( Scatter(x=tcs, y=Lmp, name="LP" ), 2, 1)
# fig.append_trace( Scatter(x=tcs, y=ws), 3, 1)
# fig.append_trace( Scatter(x=tcs, y=Ds), 4, 1)

# offline.plot( fig )
