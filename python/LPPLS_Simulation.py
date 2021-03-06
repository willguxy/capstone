# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 23:05:34 2017

@author: albert

Use synthetic data to unit test the code
"""

from utils import simulate_lppls
from LPPLSDensity import LPPLSDensity

import numpy as np

import datetime
import shutil
import pickle

import plotly.offline as offline
from plotly.graph_objs import Layout, Contour, Scatter, Figure


np.seterr(all="raise")


sample_sizes = np.arange(100, 710, 25)
cob_date = "2017-01-01"
delta_t = 100
lm_all = []
keep_all = []
density = LPPLSDensity()
for length in sample_sizes:
    data = simulate_lppls(delta_t, length)
    F2s, Lm, ms, ws, Ds, tcs, keep = density.get_density(data.time.as_matrix(),
                                                         data.price.as_matrix(),
                                                         datetime.datetime.strptime(cob_date, "%Y-%m-%d").date(),
                                                         delta_t)
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

layout = Layout(
    yaxis=dict(domain=[0.34, 1]),
    yaxis2=dict(domain=[0, 0.30]),
    shapes=[
        {"x0": cob_date, "y0": sample_sizes[0], "x1": cob_date, "y1": sample_sizes[-1], "line": {"color": "red"}},
        {"x0": tcs[9], "y0": sample_sizes[0], "x1": tcs[9], "y1": sample_sizes[-1], "line": {"color": "lime"}},
    ]
)

real_price = simulate_lppls(delta_t=-150 + delta_t, length=len(tcs))

plots = list()
plots.append(Contour(z=lm_all, y=sample_sizes, x=tcs, colorscale=my_color, showscale=False))
plots.append(Contour(z=keep_all, y=sample_sizes, x=tcs, colorscale=filter_color, showscale=False,
                     opacity=0.5, contours=dict(showlines=False)))
plots.append(Scatter(y=real_price.price, x=tcs, yaxis="y2", name="Bitcoin", line=dict(color="black")))

fig = Figure(data=plots, layout=layout)

offline.plot(fig)

shutil.copy("temp-plot.html", "./simulation/plot-" + cob_date + "-d" + str(delta_t) + ".html")
pickle.dump({"lm": lm_all, "size": sample_sizes, "time": tcs, "price": real_price, "keep": keep_all},
            open("output/saved-" + cob_date + "-d" + str(delta_t) + ".pkl", "wb"))
