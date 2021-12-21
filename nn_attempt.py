"""
Some 4am NN abomination made by scouring internet blogs.
Testing is a pain in the ass because I couldn't get tensorflow-gpu working.
Even with a aparticularly powerful CPU it takes 5 mins to train the model.
Maybe I should save the model and load it in future.

http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/
It is suggested that for an n-dimensional non linearly seperable dataset, the problem must be solved using n+1
dimensions. To achieve this some relation is commented on that for N hidden layers you may achieve N dimensions.
The page makes me believe the best I may get from the NN is some "good enough" transform of the entire dataset to
~some location, reaching a local minima that classifies everything as class A or B with no nuance. 

I will just copy everyone else and use ReLu for this many parameters, I think.

@author: Lucas Wilson
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import InputSpec

from tensorflow import keras
from tensorflow.python.client import device_lib

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.utils import class_weight

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

np.random.seed(0)

#Load price data and extract into data set and classifications 
price_data = pd.read_csv("Processed_Data.csv")[1:] #First data point has incorrect SMA 
features = ["RSI", "BB_Upper", "BB_SMA", "BB_Lower", "SMA50", "SMA100", "SMA200"]
classes = ["Sell", "Buy"]
X = price_data[features].values
y = price_data["Target"].values

scaler = StandardScaler().fit(X)
X = scaler.transform(X)
np.random.shuffle(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.3,
    random_state = 47
)

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(zip(np.unique([0, 1]), class_weights))

model = Sequential()
model.add(Dense(3, activation="tanh", input_shape=(7,) ))
model.add(Dense(3, activation="tanh", ))
model.add(Dense(1, activation="sigmoid"))

model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.SGD(
        learning_rate=.0002,
        momentum=.00002
    ),
    metrics=["accuracy"]
)

history = model.fit(
    X_train,
    y_train,
    validation_data = (X_test, y_test),
    class_weight = class_weights,
    epochs=250,
    verbose=0,
    batch_size=3
)

price_data.set_index("Date", inplace=True, drop=False)
#price_data.drop("Date", inplace=True)

plt.plot(price_data[["Close"]])
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

for i, row in price_data.iterrows():
    t = np.array([row[features]]).astype(np.float32)
    shit = model(t).numpy()[0][0]
    if shit < .5:
        print("SELL")
        plt.axvline(x=row["Date"], color=(1,0,0,.3))
    elif shit > .5:
        print("BUY")
        plt.axvline(x=row["Date"], color=(0,1,0, .3))

fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle("Training data", fontsize=16)
ax1.plot(history.history["loss"])
ax1.plot(history.history["val_loss"])
ax2.plot(history.history["accuracy"])
ax2.plot(history.history["val_accuracy"])
plt.xlabel("epoch")
ax1.set_ylabel("loss")
ax2.set_ylabel("accuracy")
plt.show()
