
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



from candlestick import atrseries,get_keltner_signals, get_engulf_signals, Signal, candle_df

candles = pd.read_csv("XBTUSD-1m-data.csv", sep=',')
candles = candle_df(candles, 1440 * 31)

def backtest_strategy(candleamount = 1440 * 31, capital = 1000, signal_params = {'keltner': True, 'engulf':True,  'kperiod':40, 'ksma':True, 'atrperiod':30, 'ignoredoji':False, 'engulfthreshold': 1, 'trade':"dynamic"}): #trade= long, short, dynamic
    atr = pd.Series
    signals = pd.DataFrame()
    kperiod = signal_params['kperiod']
    ksma = signal_params['ksma']
    atrperiod = signal_params['atrperiod']
    trade = signal_params['trade']
    ignoredoji = signal_params['ignoredoji']
    engulfthreshold = signal_params['engulfthreshold']
    posmult = signal_params['posmult']

    if(signal_params['keltner'] == True):
       signals['keltner'] = get_keltner_signals(candles, candleamount=candleamount, ma=kperiod, sma=ksma)
       #print("KELTNER SIGNALS", signals.groupby('keltner'))

    if(signal_params['engulf'] == True):
        signals['engulf'] = get_engulf_signals(candles, candleamount=candleamount, threshold=engulfthreshold)

    atr=atrseries(candles, period=atrperiod)
    visual_data = pd.DataFrame(columns= ['timestamp', 'capital'])
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
        long=True
        short=True
        if(trade=="long"):
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
                    position_amount += posmult*static_position_amount #we only get up to this point if our position is positive
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
                    position_amount -= posmult*static_position_amount #we only get up to this point if our position is negative
                print("####### SHORT ENTRY ########")
                print("Entry price:", entry_price)
                print("Stop loss:", stopPrice)
                print("Current position:", position_amount)
                print("############################")
                have_pos = True
                fee = abs(position_amount*0.00075)
                capital -= fee
        visual_data.loc[idx] = [candles.loc[idx,'timestamp'], capital]


    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    backtestfile = Path("Backtest",str(atrperiod) + str(kperiod) + str(ksma) + ".txt")
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
    f.write("\nSignals Used: ")
    f.write("Keltner:" + str(signal_params['keltner']) + ", Engulf:" + str(signal_params['engulf']))
    f.write("\nPosition multiplier: ")
    f.write(str(posmult))
    f.write("\nATR Period: ")
    f.write(str(atrperiod))
    f.write("\nKeltner Period: ")
    f.write(str(kperiod))
    f.write("\nKeltner SMA (EMA if false): ")
    f.write(str(ksma))
    f.write("\nIgnore Doji: ")
    f.write(str(ignoredoji))
    f.write("\nEngulfing Threshold: ")
    f.write(str(engulfthreshold))
    f.write("\nTrade Type: ")
    f.write(trade)
    f.write('\n---------------------------\n')
    visual_data.to_csv('VISUAL DATA')

    visualize_trades(visual_data, signal_params)

    return [signal_params, capital]

def visualize_trades(df):
    import matplotlib.pyplot 
    from matplotlib import pyplot as plt

    list_of_datetimes = df['timestamp'].tolist()
    list_of_datetimes = [t[:-6] for t in list_of_datetimes]
    l = [datetime.strptime(t, "%Y-%m-%d %H:%M:%S") for t in list_of_datetimes]
    values = df['capital'].tolist()
    dates = matplotlib.dates.date2num(l)
    matplotlib.pyplot.plot_date(dates, values,'-b')
    plt.savefig('foo.png')




atrperiod_v = [5,50]
kperiod_v = [15,30]
ksma_v = [True]
keltner_v = [True]
engulf_v = [True]
ignoredoji_v = [True,False]
trade_v = ['dynamic', 'long']
posmult_v = [2, 4, 8]
engulfthreshold_v = [.5, .75 , 1]

a = [atrperiod_v, kperiod_v, ksma_v, keltner_v, engulf_v, ignoredoji_v, trade_v, posmult_v, engulfthreshold_v]
combinations = list(itertools.product(*a))
atrperiod_v = [l[0] for l in combinations]

atrperiod_v = []
for l in combinations:
    atrperiod_v.append(l*3)

kperiod_v = [l[1] for l in combinations]
ksma_v = [l[2] for l in combinations]
params_to_try = [ {'keltner':l[3] , 'engulf':l[4],  'kperiod':l[1], 'ksma':l[2], 'atrperiod':l[0], 'ignoredoji':l[5], 'engulfthreshold': l[8], 'trade':l[6], 'posmult':l[7]} for l in combinations]
#{'keltner': True, 'engulf':True,  'kperiod':40, 'ksma':True, 'atrperiod':30, 'ignoredoji':False, 'engulfthreshold': 1, 'trade':"dynamic"}
params_to_try = [{'keltner': True, 'engulf': True, 'kperiod': 30, 'ksma': True, 'atrperiod': 5, 'ignoredoji': True, 'engulfthreshold': 1, 'trade': 'dynamic', 'posmult': 32}]
#atrperiod_v = [5,10,20,30]
#kperiod_v = [10,20,30,40]
#ksma_v = [True, True, True, True]
#results = pool.uimap(lambda atrperiod, kperiod, ksma, : backtest_strategy(atrperiod=atrperiod, kperiod=kperiod, ksma=ksma), atrperiod_v, kperiod_v, ksma_v)
#backtest_strategy()

#create multithread pool w/ number of threads being number of combinations
print("thread amount:", len(params_to_try))
#pool = ThreadPool(len(params_to_try))

backtest_strategy(signal_params=params_to_try[0])
#results = pool.uimap(lambda signal_params, : backtest_strategy(signal_params=signal_params), params_to_try)

#print("THE BEST SIGNALS ARE", max(list(results), key=lambda x:x[1]))