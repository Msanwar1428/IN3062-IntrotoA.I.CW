from Preprocessing import process_dataframe
import pandas as pd

SPY_raw_data = pd.read_csv("SPY_OHLC.csv")
SPY_raw_data = SPY_raw_data[["Close"]]

process_dataframe(SPY_raw_data) #Acts on a ref to modify in place
#may be slower than acting on a copy, TODO: compare

SPY_processed = SPY_raw_data[200:] #SMA200 is invalid before the 200th period

#build training set from random samples of the processed index data
training_set_size = 1000

training_set = SPY_processed.copy()
training_set = training_set.sample(n=training_set_size)
training_set["Target"] = 0.0

for row in training_set.index:
    cost = training_set.at[row, "Close"]
    best = cost.copy()
    worst = cost.copy()
    for x in range(30):
        try:
            price = SPY_processed.at[row+x, "Close"]
            if price > best:
                best = price.copy()
            elif price < worst:
                worst = price.copy()
        except:
            pass

    d_best = best - cost
    d_worst = cost - worst #-1 * (worst-cost) 

    training_set.at[row, "Target"] = int(d_best > d_worst) #More profit from buy than sell 

training_set[["BB_Lower"]] = training_set[["BB_Lower"]].div(training_set[["Close"]].values)
training_set[["BB_SMA"]] = training_set[["BB_SMA"]].div(training_set[["Close"]].values)
training_set[["BB_Upper"]] = training_set[["BB_Upper"]].div(training_set[["Close"]].values)
training_set[["SMA50"]] = training_set[["SMA50"]].div(training_set[["Close"]].values)
training_set[["SMA100"]] = training_set[["SMA100"]].div(training_set[["Close"]].values)
training_set[["SMA200"]] = training_set[["SMA200"]].div(training_set[["Close"]].values)

training_set.drop("Close", axis=1, inplace=True)
training_set.to_csv("Processed_Data.csv")
