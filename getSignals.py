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

def saveATR(candleamount, periods=[], fillna=True):
    for i in periods:
        df = ind.atrseries(candles, candleamount, i)
        print(df)
        df.to_csv("IndicatorData//ATR//" + "p" + str(i), mode='w')

def computeKeltner(candleamount, kperiod, ksma):
    df = pd.DataFrame()
    df = ind.get_keltner_bands(candles, candleamount=candleamount, kperiod=kperiod, ksma=ksma)
    return(df)

def saveKeltners(candleamount, kperiods=[], ksmas=[]):
    a = [kperiods, ksmas]
    combinations = list(itertools.product(*a))
    print(combinations)
    for i in combinations:
        df = computeKeltner(candleamount, i[0], i[1])
        print(df)
        df.to_csv("IndicatorData//Keltner//" + "kp" + str(i[0]) + '_sma' + str(i[1]), mode='w')

saveKeltners(100, [10,1], [True, False])
saveATR(100, [1,20,30])