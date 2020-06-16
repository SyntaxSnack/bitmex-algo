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
#import tracemalloc
import time
from pathos.threading import ThreadPool
import itertools

class Signal(Enum):
    WAIT = 0
    BUY = 1
    SELL = 2

XBTUSD = .5
ETHUSD = .05
MIN_IN_DAY = 1440
candle_data = []
signals=[]

prev_row = None

def candle_df_thread(index, data):
    if data[2] < data[3]:
        type = "green"
    elif data[2] > data[3]:
        type = "red"
    else:
        type = "abs_doji"
    #append size
    candle_data = [data[0], data[1], data[2],data[3], data[4], abs(data[2]-data[3]), type]
    return(candle_data)

def candle_df(candles, candleamount):
    print("candle_df")
    # iterate over rows with iterrows()
    cpool = ThreadPool()
    #for index, data in candles.tail(candleamount).iterrows():
        #candle_df_thread(index, data)
    indices = candles.tail(candleamount).index.values.tolist()
    data = candles.tail(candleamount).values.tolist()
    results = cpool.uimap(candle_df_thread, indices, data)
    print("Computing candlestick dataframe for given params with candles multithreaded...")
    result = list(results)
    print(results)
    return(result)

#realtime func
def engulfingsignals(curr_row, threshold, ignoredoji):
    global prev_row
    if curr_row[6] == prev_row[6]: #candle type stays the same
        signals = Signal.WAIT
    elif ((curr_row[5] * threshold) > prev_row[5]) and (ignoredoji == False or prev_row[5] >= XBTUSD): # candle is opposite direction and larger
        if curr_row[6] == "red":
            signals = Signal.SELL
        elif curr_row[6] == "green":
            signals = Signal.BUY
        else:
            signals = Signal.WAIT
    else:
        signals = Signal.WAIT
    prev_row = curr_row
    return(signals)

def keltnersignals(row):
    if row.loc['lband'] == 1.0:
        return Signal.BUY
    elif row.loc['hband'] == 1.0:
        return Signal.SELL
    else:
        return Signal.WAIT

def atrseries(candles, candleamount, period, fillna=True):
    candles = candles.tail(candleamount)
    atr = ta.volatility.AverageTrueRange(candles["high"], candles["low"], candles["close"], n=period, fillna=fillna)
    series = pd.Series(dtype=np.uint16)
    series = atr.average_true_range()
    return(series)

#back-test on the series extrapolated from price data
def get_engulf_signals(e_candles, candleamount, params):
    global signals
    global prev_row
    prev_row = e_candles[0]
    threshold = np.repeat(params[0], len(e_candles))
    ignoredoji = np.repeat(params[1], len(e_candles))
    results = []
    for i in range(len(e_candles)):
        #print(engulfingsignals(e_candles[i], threshold[i], ignoredoji[i]))
        results.append(engulfingsignals(e_candles[i], threshold[i], ignoredoji[i]))
    #results = epool.uimap(engulfingsignals, e_candles, threshold, ignoredoji)
    print("Computing engulfing signals with given params for all candles multithreaded...")
    result = list(results)
    return(result)

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
    for i,row in kseries.iterrows():
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