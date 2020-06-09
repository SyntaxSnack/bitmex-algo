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
from functools import partial
from datetime import datetime
import indicators as ind
import os.path
from os import path
from backtest import candles

t_symbol = None
t_e_candles = pd.DataFrame()
t_candleamount = None

def saveATR(candleamount, params=[], fillna=True, symbol='XBTUSD'):
    for i in params:
        if path.exists('IndicatorData//' + t_symbol + '//ATR//' + "p" + str(i) + '.csv'):
            return 
        df = ind.atrseries(candles, candleamount, i)
        print(df)
        df.to_csv('IndicatorData//' + symbol + '//ATR//' + "p" + str(i) + ".csv", mode='w')
    return
def saveKeltnerBands(candleamount, params=[], symbol='XBTUSD'):
    for i in params:
        if path.exists('IndicatorData//' + symbol + '//Keltner//' + "BANDS_kp" + str(i[0]) + "_sma" + str(i[1]) + '.csv'):
            continue
        df = ind.get_keltner_bands(candles, candleamount=candleamount, kperiod=i[0], ksma=i[1])
        print(df)
        df.to_csv('IndicatorData//' + symbol + '//Keltner//' + "BANDS_kp" + str(i[0]) + "_sma" + str(i[1]) + '.csv', mode='w')
    return
def saveKeltnerSignals(candleamount, params=[], symbol='XBTUSD'):
    for i in params:
        if path.exists('IndicatorData//' + symbol + '//Keltner//' + "SIGNALS_kp" + str(i[0]) + "_sma" + str(i[1]) + '.csv'):
            continue
        signals =  ind.get_keltner_signals(candles, candleamount=candleamount, kperiod=i[0], ksma=i[1])
        df = pd.Series(signals, dtype=object)
        print(df)
        df.to_csv('IndicatorData//' + symbol + '//Keltner//' + "SIGNALS_kp" + str(i[0]) + "_sma" + str(i[1]) + '.csv', mode='w', index=False)
    return

def saveEngulf_thread(params):
    if path.exists('IndicatorData//' + t_symbol + '//Engulfing//' + "SIGNALS_t" + str(params[0]) + "_ignoredoji" + str(params[1]) + '.csv'):
        return
              
    signals = ind.get_engulf_signals(t_e_candles, t_candleamount, params)
    df = pd.Series(signals, dtype=object)
    df.to_csv('IndicatorData//' + t_symbol + '//Engulfing//' + "SIGNALS_t" + str(params[0]) + "_ignoredoji" + str(params[1]) + '.csv', mode='w', index=False)
    return("thread-done")

def saveEngulfingSignals(candleamount, params=[], symbol='XBTUSD'):
    global t_e_candles
    global t_symbol
    global t_candleamount
    t_e_candles = ind.candle_df(candles, candleamount)
    t_symbol = symbol
    t_candleamount = candleamount
    epool = ThreadPool()
    results = epool.uimap(saveEngulf_thread, params)
    print("Computing engulfing signals for all params multithreaded...")
    #DO NOT REMOVE THIS PRINT, IT IS NEEDED TO FINISH THE MULTITHREAD
    result = list(results)
    print(result)

    return(result)

#Examples
#saveKeltnerBands(100, [10,1], [True, False])
#saveATR(100, [1,20,30])