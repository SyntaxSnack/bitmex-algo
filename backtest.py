
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


from candlestick import atrseries,get_keltner_signals, get_engulf_signals, Signal
'''
def backtest_strategy(candleamount = 1440, signals_to_use = {'keltner': True, 'engulf':True}, kperiod=40, ksma=True, atrperiod=30, capital = 1000, ignoredoji = False, engulfthreshold = 1, trade="dynamic"): #trade= long, short, dynamic
    atr = pd.Series
    signals = pd.DataFrame()

    if(signals_to_use['keltner'] == True):
       signals['keltner'] = get_keltner_signals(candles, candleamount=candleamount, ma=kperiod, sma=ksma)
       #print("KELTNER SIGNALS", signals.groupby('keltner'))

    if(signals_to_use['engulf'] == True):
        signals['engulf'] = get_engulf_signals(candles, candleamount=candleamount)

    atr=atrseries(candles, period=atrperiod)

    entry_price = 0
    profit = 0
    signals.to_csv('signals')
    position_size = .1
    position_amount = capital * position_size
    static_position_amount = capital * position_size
    fee = position_amount * 0.00075
    have_pos = False
    stop_loss = .1
    stop=False
    stopPrice=0
    stopType="atr"
    for idx, data in signals.head(candleamount).iterrows():
        if(trade=="dynamic"): #write an algo for this later
            long=True
            short=True
        elif(trade=="long"):
            long=True
            short=False
        elif(trade=="short"):
            long=False
            short=True
        if (entry_price != 0 and stopPrice != 0):
            if(((position_amount > 0) and (candles.loc[idx,'open'] < stopPrice)) or (((position_amount < 0)) and (candles.loc[idx,'open'] > stopPrice))):
                stop=True
                print("!!!!!! STOP PRICE HIT !!!!!!")
                print("ATR stop threshold: ",atr[idx-1]*50)
        if((all([v == Signal.BUY for v in data])) or (stop and position_amount < 0)):
            if((short and have_pos) and (position_amount < 0)):
            #short is only a constant parameter to determine if we are shorting at all
            #position_amount determines if we are currently short or long
                profit = position_amount * ((candles.loc[idx,'open'] - entry_price)/entry_price)
                capital += profit
                capital -= fee
                position_amount = 0
                print("######## SHORT EXIT ########")
                print("Exit price:", candles.loc[idx,'open'])
                print("Turnover:", profit - fee*2)
                print("############################")
                #100 * (200-400/400) = 100
                entry_price = 0
                have_pos = False
                stop = False
            elif(candles.loc[idx+1,'open'] > entry_price):
                entry_price = candles.loc[idx+1,'open']
                if(stopType == "atr"): #and if we already have a position, our stop moves up to reflect second entry
                    stopPrice = entry_price - atr[idx]*50
                elif(stopType == "perc"):
                    stopPrice = entry_price * (1-stop_loss)
                if(have_pos == False):
                    position_amount = static_position_amount
                else:
                    position_amount += 4*static_position_amount #we only get up to this point if our position is positive
                fee = position_amount*0.00075
                capital -= fee
                print("######## LONG ENTRY ########")
                print("Entry price:", entry_price)
                print("Stop loss:", stopPrice)
                print("Current position:", position_amount)
                print("############################")
                have_pos = True
        elif(all([v == Signal.SELL for v in data]) or (stop and position_amount > 0)):
            if((long and have_pos) and (position_amount > 0)):
                profit = position_amount * ((candles.loc[idx,'open'] - entry_price)/entry_price)
                capital += profit
                capital -= fee
                position_amount = 0
                print("######### LONG EXIT ########")
                print("Exit price:", candles.loc[idx,'open'])
                print("Turnover:", profit - fee*2)
                print("############################")
                #100 * (200-400/400) = 100
                entry_price = 0
                have_pos = False
                stop = False
            elif(short and ((candles.loc[idx+1,'open'] < entry_price) or entry_price==0)): #only add to position if original position is in profit!
                entry_price =  candles.loc[idx+1,'open']
                if(stopType == "atr"):
                    stopPrice = entry_price + atr[idx]*50
                elif(stopType == "perc"):
                    stopPrice = entry_price * (1+stop_loss)
                if(have_pos == False):
                    position_amount = -1*static_position_amount
                else:
                    position_amount -= 4*static_position_amount #we only get up to this point if our position is negative
                print("####### SHORT ENTRY ########")
                print("Entry price:", entry_price)
                print("Stop loss:", stopPrice)
                print("Current position:", position_amount)
                print("############################")
                have_pos = True
                fee = abs(position_amount*0.00075)
                capital -= fee

    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    backtestfile = Path("Backtest",time + ".txt")
    f = open(backtestfile, "a")
    f.write('\n---------------------------')
    f.write('\n---- BACKTEST COMPLETE ----')
    f.write("\nBacktest time (days):\n")
    f.write(str(candleamount/1440))
    f.write("\nFinal capital:\n")
    f.write(str(capital))
    f.write("\nTotal profit:\n")
    f.write(str(capital-1000))
    f.write("\n----- Parameters used -----")
    f.write("\nIgnore Doji: ")
    f.write(str(ignoredoji))
    f.write("\nEngulfing Threshold: ")
    f.write(str(engulfthreshold))
    f.write("\nTrade Type: ")
    f.write(trade)
    f.write('\n---------------------------\n')

    return("VARIABLES TESTED FOR",candleamount)
    #for i in candles.head(candleamount).iterrows():
    return signals

#define data to test with
candles = pd.read_csv("XBTUSD-1m-data.csv", sep=',')

#create multithread pool
pool = ThreadPool()

from functools import partial
#mapfunc = partial(backtest_strategy, 'atrperiod')
'''
'''
POSSIBLE ARGUMENTS TO BACKTEST WITH:
format variables like this to multi-thread several variations at once var = [value1, 2, 3, etc]
    candleamount = MIN_IN_DAY
    signals_to_use = {'keltner': True, 'engulf':True}
    kperiod=40
    ksma=True
    atrperiod=30
    capital = 1000
    ignoredoji = False
    engulfthreshold = 1
    trade="dynamic"
#indices of value in tuples [] correspond to each other
'''
'''
atrperiod_v = [5,10,20,30]
kperiod_v = [10,20,30,40]
ksma_v = [True]




a = [atrperiod_v, kperiod_v, ksma_v]
combinations = list(itertools.product(*a))
atrperiod_v = [l[0] for l in combinations]
kperiod_v = [l[1] for l in combinations]
ksma_v = [l[2] for l in combinations]

print(atrperiod_v)
print(kperiod_v)
print(ksma_v)
#atrperiod_v = [5,10,20,30]
#kperiod_v = [10,20,30,40]
#ksma_v = [True, True, True, True]
results = pool.imap(lambda atrperiod, kperiod, ksma, : backtest_strategy(atrperiod=atrperiod, kperiod=kperiod, ksma=ksma), atrperiod_v, kperiod_v, ksma_v)
print(list(results))
    #terminate process on key press
    #stop_char=""
    #while stop_char.lower() != "q":
    #    stop_char=input("Enter 'q' to quit ")
    #print("terminate process")
    #p.terminate()
    '''