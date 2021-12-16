import pandas as pd
import matplotlib.pyplot as plt
import math

RSI_PERIODS = 14
BB_PERIODS = 20

def raw_RSI_data(df, index):
    data = df[index-RSI_PERIODS-1:index+1]

    deltas = data.pct_change().multiply(100).rename(columns={"Close" : "Chg%"})[1:].reset_index(drop=True)

    mask = deltas["Chg%"] < 0
    deltas["Gain"] = deltas["Chg%"].mask(mask)
    deltas["Loss"] = deltas["Chg%"].mask(~mask).abs()

    return deltas

def raw_RSI(df, index):
    """
    Calculates the non smoothed RSI for a given index, using RSI_PERIODS periods
    https://school.stockcharts.com/doku.php?id=technical_indicators:relative_strength_index_rsi
    """
    deltas = raw_RSI_data(df, index)

    first_gain = deltas["Gain"].sum() 
    first_loss = deltas["Loss"].sum()

    RS = ( first_gain / RSI_PERIODS ) / ( first_loss / RSI_PERIODS )
    return 100 - 100 / (1 + RS)

def RSI(df, index):
    """
    Calculates smoothed RSI for given index, with RSI_PERIODS periods
    """
    if index <= RSI_PERIODS:
        raise Exception("RSI insufficient past data")
    elif index == RSI_PERIODS+1:
        return raw_RSI(df, index)
    
    deltas = raw_RSI_data(df, index)
    
    current_delta = df.iloc[index-1:index+1].pct_change()[1:].reset_index(drop=True).iloc[0,0].item()
    curr_gain = max(0, current_delta)
    curr_loss = abs(min(0, current_delta))

    prev_avg_gain = deltas["Gain"].sum() / RSI_PERIODS
    prev_avg_loss = deltas["Loss"].sum() / RSI_PERIODS

    avg_gain = ( prev_avg_gain * (RSI_PERIODS - 1) + curr_gain ) / RSI_PERIODS
    avg_loss = ( prev_avg_loss * (RSI_PERIODS - 1) + curr_loss ) / RSI_PERIODS
    
    RS = avg_gain / avg_loss
    return 100 - 100 / (1 + RS)

def raw_SMA(df):
    return df[["Close"]].sum() / len(df.index)

def SMA(df, index, periods):
    data = df[index-periods: index]
    return raw_SMA(data)

def calc_bollinger_bands(df, index):
    data = df[index-BB_PERIODS: index]
    data["Deviation"] = 0.0
    data["DeviationSq"] = 0.0 #Squared

    data = data.reset_index(drop=True)

    mean = raw_SMA(data)
    for row in data.index:
        dev = data.iloc[row-1, 0] - mean
        data.at[row, "Deviation"] = dev
        data.at[row, "DeviationSq"] = dev*dev
        
    AvgDevSq = data[["DeviationSq"]].sum() / BB_PERIODS
    StdDev = math.sqrt(AvgDevSq)

    BBSMA = SMA(df, index, BB_PERIODS)["Close"]

    BB_Lower = BBSMA - (StdDev * 2)
    BB_Upper = BBSMA + (StdDev * 2)
    
    return (BB_Lower, BBSMA, BB_Upper)

def process_dataframe(df):
    pd.options.mode.chained_assignment = None  # default='warn'
    
    df["RSI"] = 0.0
    df["BB_Lower"] = 0.0
    df["BB_SMA"] = 0.0
    df["BB_Upper"] = 0.0
    df["SMA50"] = 0.0
    df["SMA100"] = 0.0
    df["SMA200"] = 0.0
    for row in df.index:
        if row > RSI_PERIODS:
            df.at[row, "RSI"] = RSI(df, row)
        if row > BB_PERIODS:
            BB_Data = calc_bollinger_bands(df, row)
            df.at[row, "BB_Lower"] = BB_Data[0]
            df.at[row, "BB_SMA"] = BB_Data[1]
            df.at[row, "BB_Upper"] = BB_Data[2]
        if row > 50:
            df.at[row, "SMA50"] = SMA(df, row, 50)
        if row > 100:
            df.at[row, "SMA100"] = SMA(df, row, 100)
        if row > 200:
            df.at[row, "SMA200"] = SMA(df, row, 200)#Takes a while to process
            

    pd.options.mode.chained_assignment = 'warn'

