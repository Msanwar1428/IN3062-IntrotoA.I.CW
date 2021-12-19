"""
Plots the relation between features in the feature set by rendering the features against eachother in 2d in a 7x7 grid.
The data is standardized per column. May not be performant vs standardizing the entire feature set sans target and index.
Same result though. 

@author: Lucas Wilson
@date: 19/12/2021
"""


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.preprocessing import StandardScaler

feature_names = [
    "RSI",
    "BB_Upper",
    "BB_SMA",
    "BB_Lower",
    "SMA50",
    "SMA100",
    "SMA200",
    "Target"
]

price_data = pd.read_csv("Processed_Data.csv")[200:][feature_names]
#First 200 data rows will not have calculated SMA200
#Earlier rows may not have other features calculated either
#@Author: Lucas Wilson

def scale(feature_name):
    if feature_name == "Target":
        return
    scaler = StandardScaler().fit(price_data[feature_name].values.reshape(-1, 1))
    price_data[feature_name] = scaler.transform(price_data[feature_name].values.reshape(-1,1))

for fn in feature_names:
    scale(fn)

fig, axes = plt.subplots(7,7)
fig.suptitle("Data Examination", fontsize=16)

for x in range(7):
    for y in range(7):
        """Iterate the selected features and plot them in 2x2 charts
        to attempt to interpret the relevance of the features on the solution
        @Author: Lucas Wilson"""

        pdv = price_data.values
        
        ax_id = x * 7 + y
        ax = axes[x][y]

        x_data = pdv[:,x]
        y_data = pdv[:,y]
        classification = pdv[:,7]
        colors = []
        for k in classification:
            if k > .99:
                colors.append((0,1,0))
            else:
                colors.append((1,0,0))

        ax.set_xlabel(feature_names[x])
        ax.set_ylabel(feature_names[y])

        ax.scatter(x_data, y_data, marker="o", color=colors, s = 1)

plt.show()
