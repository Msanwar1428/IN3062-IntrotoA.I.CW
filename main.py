"""
Load the training data, divide into 70/30 distribution
Quick & Simple test of the model 

TODO: 50/25/25 distribution for train/test/holdout sets

@author: Lucas Wilson
"""

import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split 

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
    if x != "Target" and x != "Date": 
        result.append(x)
        #clip useless columns from the data set 

X = price_data[result].values
y = price_data["Target"].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = .3,
    random_state=51
)

model = SVC(
    gamma=.01,
    decision_function_shape="ovo", #binary classification
    kernel = "rbf", #expecting necessity to cast to higher dim space to solve
    C = 1, #A fairly low value of C that should avoid overfitting
    class_weight = "balanced" #attempt to solve for 64.6% buy target imbalance
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

output = pd.DataFrame(data=np.c_[y_test, y_pred])

cm = confusion_matrix(y_test, y_pred)
cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]


#draw price series and prediction lines
price_data["Date"] = pd.to_datetime(price_data["Date"], format="%Y-%m-%d")
price_data.set_index("Date", drop=False, inplace=True)
#price_data.drop("Date", inplace=True)
print(price_data)


ax = price_data[["Close"]].plot(x_compat=True)
ax.xaxis.set_major_locator(mdates.MonthLocator())

TP = 0
FP = 0

for idx, row in price_data.iterrows():
    #print(price_data[row:row+1])
    pred = model.predict([row[result]])
    if pred != row.loc["Target"]:
        #print("wrong")
        if pred > .99:
            ax.axvline(x=row["Date"], color=(0,1,0))
            FP+=1
        else:
            ax.axvline(x=row["Date"], color=(1,0,0))
        pass
    else:
        if pred > .99:
            TP+=1
        #print("right")
        pass

print(TP / (TP + FP))

plt.figure(2)
plot_confusion_matrix(cm_normalized, ["Sell", "Buy"], title = "Normalized Confusion")
plt.show()



