# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 14:51:06 2017

@author: alber
"""

import pickle
from plotly import tools
import plotly.offline as offline
from plotly.graph_objs import Layout, Contour, Scatter, Figure

data = pickle.load(open("saved.pkl", "rb"))
lm_all = data["lm"]
tcs = data["time"]
sample_sizes = data["size"]
keep_all = data["keep"]
prices = data["price"]

cob_date = "2017-05-25"

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

plots = list()
plots.append(Contour(z=lm_all, y=sample_sizes, x=tcs, colorscale=my_color, showscale=False))
plots.append(Contour(z=keep_all, y=sample_sizes, x=tcs, colorscale=filter_color, showscale=False,
                     opacity=0.5, contours=dict(showlines=False)))
plots.append(Scatter(y=prices.price, x=tcs, yaxis="y2", name="Bitcoin", line=dict(color="black")))

fig = Figure(data=plots, layout=layout)

offline.plot(fig)

