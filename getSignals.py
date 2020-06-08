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

candles = pd.read_csv("XBTUSD-1m-data.csv", sep=',')
print(candles)

def saveATR(candleamount, params=[], fillna=True, symbol='XBTUSD'):
    for i in params:
        df = ind.atrseries(candles, candleamount, i)
        print(df)
        df.to_csv("IndicatorData//" + symbol + "//ATR//" + "p" + str(i) + '.csv', mode='w')

def saveKeltnerBands(candleamount, params=[], symbol='XBTUSD'):
    for i in params:
        df = ind.get_keltner_bands(candles, candleamount=candleamount, kperiod=i[0], ksma=i[1])
        print(df)
        df.to_csv("IndicatorData//" + symbol + "//Keltner//" + "BANDS_kp" + str(i[0]) + '_sma' + str(i[1]), mode='w')

def saveKeltnerSignals(candleamount, params=[], symbol='XBTUSD'):
    for i in params:
        signals = ind.get_keltner_signals(candles, candleamount=candleamount, kperiod=i[0], ksma=i[1])
        df = pd.Series(signals)
        print(df)
        df.to_csv("IndicatorData//" + symbol + "//Keltner//" + "SIGNALS_kp" + str(i[0]) + '_sma' + str(i[1]), mode='w')

def saveEngulfingSignals(candleamount, params=[], symbol='XBTUSD'):
    e_candles = ind.candle_df(candles, candleamount)
    for i in params:
        print(type(i[0]))
        signals = ind.get_engulf_signals(e_candles, candleamount, threshold=i[0], ignoredoji=i[1])
        df = pd.Series(signals)
        print(df)
        df.to_csv("IndicatorData//" + symbol + "//Engulfing//" + "SIGNALS_kp" + str(i[0]) + '_sma' + str(i[1]), mode='w')

#Examples
#saveKeltnerBands(100, [10,1], [True, False])
#saveATR(100, [1,20,30])