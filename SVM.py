"""
Load the training data, divide into 70/30 distribution
Quick & Simple test of the model
Model tuned by CVGridSearch to tune hyperparameters.

TODO:
evaluate 50/25/25 distribution for train/test/holdout sets
iterate gridsearch at increasing fidelity levels


@author: Lucas Wilson
"""

import pandas as pd
import numpy as np

from sklearn.experimental import enable_halving_search_cv

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import HalvingGridSearchCV

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#method to plot confusion matrices
def plot_confusion_matrix(cm, names, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Plots confusion matrix
    @Author: jacob
    source IN3062 lecture 3 exercise solutions
    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(fraction=0.05)
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

try:
    price_data = pd.read_csv("Processed_Data.csv")
except FileNotFoundError:
    print("Please run GenTrainingData.py to generate the Processed_Data.csv file")
    exit()

#price_data.reset_index(inplace=True, drop=True)
#price_data = price_data.reindex(np.random.permutation(price_data.index))
#format and shuffle the index, index does not carry relevant data

result = []
for x in price_data.columns:
    if x != "Target" and x != "Date" and x != "Close" and x != "Unnamed: 0": 
        result.append(x)
        #clip useless columns from the data set

X = price_data[result].values
y = price_data["Target"].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = .3,
    random_state=52
)

params = {
    #"C" : [0.1, 1, 10, 100, 250, 500],
    #"gamma" : [100, 75, 50, 25, 10, 1, .1, .01, .001],
    "C" : [1, 10, 100, 250, 300, 400, 450, 600],
    "gamma" : [50, 130, 142, 150, 175, 200, 225]
}

grid = GridSearchCV(
    SVC(
        decision_function_shape="ovo", #binary classification
        kernel = "rbf", #expecting necessity to cast to higher dim space to solve
        class_weight = "balanced", #attempt to solve for 64.6% buy target imbalance,
        probability = True
    ),
    params,
    refit = True,
    verbose = 0
)

grid.fit(X_train, y_train)
print(grid.best_estimator_)

y_true = y 
y_pred = grid.predict(X)
#print(y_pred)

output = pd.DataFrame(data=np.c_[y_true, y_pred])

cm = confusion_matrix(y_true, y_pred)
cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

TP = 0
FP = 0
TN = 0
FN = 0

#draw price series and prediction lines
price_data["Date"] = pd.to_datetime(price_data["Date"], format="%Y/%m/%d")
price_data.set_index("Date", drop=False, inplace=True)
price_data.drop("Unnamed: 0", axis=1, inplace=True)

fig, (ax, ax2) = plt.subplots(2)

profits = 1.0 #0% profits
mode = ""
last_mode_price = 0

def swap_mode(new_mode, price):
    global mode, profits, last_mode_price
    if new_mode == "BUY" and mode == "SELL":
        profits *= last_mode_price/price
    elif new_mode == "SELL" and mode == "BUY":
        profits *= price/last_mode_price

    mode = new_mode
    last_mode_price = price

for idx, row in price_data.iterrows():
    pred = grid.predict_proba([row[result]])[0]
    #if pred != row.loc["Target"]:
        #print("wrong")
        #if pred > .99:
            #ax.axvline(x=row["Date"], color=(0,1,0))
            #FP+=1
        #else:
            #ax.axvline(x=row["Date"], color=(1,0,0))
            #pass
        
    #else:
    if pred[1] > .5:
        if row.Target > .99:
            TP += 1
        else:
            FP += 1
        
        if mode != "BUY":
            ax.axvline(x=row["Date"], color=(0,1,0))
            swap_mode("BUY", row["Close"])
            
    elif pred[0] > .5:
        if row.Target < .01:
            TN += 1
        else:
            FN += 1
            
        if mode != "SELL":
            ax.axvline(x=row["Date"], color=(1,0,0))
            swap_mode("SELL", row["Close"])
        #print("right")


print("Precision: " + str(TP / (TP + FP)))
print("Recall: " + str(TP / (TP + FN)))
print("PROFIT MULTIPLE: " + str(profits))

price_data["BB_SMA"] = price_data["BB_SMA"].mul(price_data["Close"])
price_data["BB_Upper"] = price_data["BB_Upper"] * price_data["Close"]
price_data["BB_Lower"] = price_data["BB_Lower"] * price_data["Close"]
price_data["SMA50"] = price_data["SMA50"] * price_data["Close"]
price_data["SMA100"] = price_data["SMA100"] * price_data["Close"]
price_data["SMA200"] = price_data["SMA200"] * price_data["Close"]

price_data[["Close"]].plot(ax=ax, x_compat=True)
price_data[["BB_Upper", "BB_Lower"]].plot(ax=ax, color=(1,0,0))
price_data[["BB_SMA"]].plot(ax=ax, color=(1, .3, 0))
price_data[["SMA50"]].plot(ax=ax)
price_data[["SMA100"]].plot(ax=ax)
price_data[["SMA200"]].plot(ax=ax)

ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax2.set_xlabel("Date")
ax2.set_ylabel("RSI")

price_data[["RSI"]].plot(ax=ax2)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

plt.figure(2)
plot_confusion_matrix(cm_normalized, ["Sell", "Buy"], title = "Normalized Confusion")

plt.show()



