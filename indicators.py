import pandas as pd
import numpy as np
from collections import Counter
from enum import Enum
import ta
from ta.utils import dropna
import asyncio
import sys
from datetime import datetime
from pathlib import Path
from pathos.multiprocessing import ProcessingPool as Pool
import tracemalloc
import time
from pathos.threading import ThreadPool
import itertools

tracemalloc.start()

class Signal(Enum):
    WAIT = 0
    BUY = 1
    SELL = 2

XBTUSD = .5
ETHUSD = .05
MIN_IN_DAY = 1440
    
def candle_df(candles, candleamount):
    candle_data = pd.DataFrame(columns=['timestamp', 'open', 'close','high', 'low', 'candle size', 'type'])
    # iterate over rows with iterrows()
    for index, data in candles.tail(candleamount).iterrows():
        #determine if candles are of opposite type
        if data['open'] < data['close']:
            type = "green"
        elif data['open'] > data['close']:
            type = "red"
        else:
            type = "abs_doji"
     #append size
        candle_data.loc[index] = [data['timestamp'], data['open'], data['close'],data['high'], data['low'], abs(data['open']-data['close']), type]
    return candle_data

#realtime func
def engulfingsignals(curr_row, prev_row, threshold = 1, ignoredoji = False):
    print(curr_row)
    if curr_row['type'] == prev_row['type']: #candle type stays the same
        return Signal.WAIT
    elif (curr_row['candle size'] * threshold) > (prev_row['candle size']) and (ignoredoji == False or prev_row['candle size'] > XBTUSD): # candle is opposite direction and larger
        if curr_row['type'] == "red":
           # print("SELLLLL")
            return Signal.SELL
        elif curr_row['type'] == "green":
           # print("BUUYYYYY")
            return Signal.BUY
        else: return Signal.WAIT
    else:
        return Signal.WAIT

def keltnersignals(row):
    #print(type(row[1]), row[1][1])
    if row.loc['lband'] == 1.0:
        return Signal.BUY
    elif row.loc['hband'] == 1.0:
        return Signal.SELL
    else:
        return Signal.WAIT

def atrseries(candles, candleamount, period, fillna=True):
    candles = candles.tail(candleamount)
    atr = ta.volatility.AverageTrueRange(candles["high"], candles["low"], candles["close"], n=period, fillna=fillna)
    series = pd.Series()
    series = atr.average_true_range()
    return(series)

#back-test on the series extrapolated from price data
def get_engulf_signals(e_candles, candleamount = MIN_IN_DAY, threshold=1, ignoredoji=False):
    #first generate a candle-series!
    #candles = candle_df(candles, candleamount)
    signals=[Signal.WAIT]
    #generate a trade signal for every candle except for the last, and store in the list we created
    prev_row = e_candles.iloc[0]
    for i,row in e_candles.tail(candleamount).iloc[1:].iterrows():
        signals.append(engulfingsignals(row, prev_row, threshold, ignoredoji))
        prev_row = row
    return signals

def keltner(candles, candleamount, kperiod, ksma):
    candles = candles.tail(candleamount)
    return(ta.volatility.KeltnerChannel(high=candles["high"], low=candles["low"], close=candles["close"], n=kperiod, fillna=True, ov=ksma))

def get_keltner_signals(candles, candleamount = MIN_IN_DAY, kperiod=10, threshold = 1, ignoredoji = False, ksma=True):
    indicator_kelt = keltner(candles, candleamount, kperiod, ksma)
    kseries = pd.DataFrame(columns=['hband', 'lband'])
    # Add Bollinger Bands features
    kseries["hband"] = indicator_kelt.keltner_channel_hband_indicator()
    kseries["lband"] = indicator_kelt.keltner_channel_lband_indicator()
    signals=[]

    for i,row in kseries.tail(candleamount).iterrows():
        signals.append(keltnersignals(row))
    return signals

def get_keltner_bands(candles, candleamount = MIN_IN_DAY, kperiod=10, threshold = 1, ignoredoji = False, ksma=True):
    indicator_kelt = keltner(candles, candleamount, kperiod, ksma)
    kseries = pd.DataFrame(columns=['hband', 'lband'])
    # Add Bollinger Bands features
    kseries["hband"] = indicator_kelt.keltner_channel_hband()
    kseries["lband"] = indicator_kelt.keltner_channel_lband()
    kseries["w"] = indicator_kelt.keltner_channel_wband()
    return(kseries)