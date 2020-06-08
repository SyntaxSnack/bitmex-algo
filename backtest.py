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
from indicators import Signal, candle_df
import getSignals as getSignals

symbol = "XBTUSD"
candles = pd.read_csv(symbol + "-1m-data.csv", sep=',')
#print(candles)
candleamount = 140
e_candles = candle_df(candles, candleamount)

def backtest_strategy(candleamount=candleamount, capital = 1000, signal_params = {'keltner': True, 'engulf':True,  'kperiod':40, 'ksma':True, 'atrperiod':30, 'ignoredoji':False, 'engulfthreshold': 1, 'trade':"dynamic"}, symbol=symbol): #trade= long, short, dynamic
    atr = pd.Series
    signals = pd.DataFrame()
    kperiod = signal_params['kperiod']
    ksma = signal_params['ksma']
    atrperiod = signal_params['atrperiod']
    trade = signal_params['trade']
    ignoredoji = signal_params['ignoredoji']
    engulfthreshold = signal_params['engulfthreshold']
    posmult = signal_params['posmult']
    candle_data = candles.tail(candleamount)

    if(signal_params['keltner'] == True):
       signals['keltner'] = pd.read_csv('IndicatorData//' + symbol + "//Keltner//" + "SIGNALS_kp" + signal_params['kperiod'] + '_sma' + signal_params['ksma'], mode='w')
       #print("KELTNER SIGNALS", signals.groupby('keltner'))

    if(signal_params['engulf'] == True):
        signals['engulf'] = get_engulf_signals(candle_data, candleamount=candleamount, threshold=engulfthreshold)

    
    atrseries = pd.read_csv('IndicatorData//' + symbol + "//ATR//" + "p" + str(atrperiod) + '.csv', sep=',')

    signal_len = len(signals.loc[0])
    candle_data = candle_data.reset_index(drop=True)
    candle_data = pd.DataFrame.join(candle_data, atrseries)
    candle_data = pd.DataFrame.join(candle_data, signals)   #COMBINE SIGNALS AND CANDLE DATA
    #print("CANDLE DATA")
    #print(candle_data)
    
    

    visual_data = pd.DataFrame(columns= ['timestamp', 'capital'])
    entry_price = 0
    profit = 0
    currentTime = datetime.now().strftime("%Y%m%d-%H%M")
    signals.to_csv('BacktestData//Signals//' + currentTime + '.csv')
    position_size = .1
    position_amount = capital * position_size
    static_position_amount = capital * position_size
    fee = position_amount * 0.00075
    have_pos = False
    stop_loss = .1
    stop=False
    stopPrice=0
    stopType="atr"
    for idx, data in candle_data.tail(candleamount).iterrows():
        long=True
        short=True
        if(trade=="long"):
            long=True
            short=False
        elif(trade=="short"):
            long=False
            short=True
        if (entry_price != 0 and stopPrice != 0):
            if(((position_amount > 0) and (data['open'] < stopPrice)) or (((position_amount < 0)) and (data['open'] > stopPrice))):
                stop=True
                print("!!!!!! STOP PRICE HIT !!!!!!")
                print("ATR stop threshold: ", data['atr']*50)
        if((all([v == Signal.BUY for v in data[-signal_len:]])) or (stop and position_amount < 0)):
            if((short and have_pos) and (position_amount < 0)):
            #short is only a constant parameter to determine if we are shorting at all
            #position_amount determines if we are currently short or long
                profit = position_amount * ((data['open'] - entry_price)/entry_price)
                capital += profit
                capital -= fee
                position_amount = 0
                print("######## SHORT EXIT ########")
                print("Exit price:", data['open'])
                print("Turnover:", profit - fee*2)
                print("############################")
                #100 * (200-400/400) = 100
                entry_price = 0
                have_pos = False
                stop = False
            elif(data['close'] > entry_price):
                entry_price = data['close']
                if(stopType == "atr"): #and if we already have a position, our stop moves up to reflect second entry
                    stopPrice = entry_price - data['atr']*50
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
        elif(all([v == Signal.SELL for v in data[-signal_len:]]) or (stop and position_amount > 0)):
            if((long and have_pos) and (position_amount > 0)):
                profit = position_amount * ((data['open'] - entry_price)/entry_price)
                capital += profit
                capital -= fee
                position_amount = 0
                print("######### LONG EXIT ########")
                print("Exit price:", data['open'])
                print("Turnover:", profit - fee*2)
                print("############################")
                #100 * (200-400/400) = 100
                entry_price = 0
                have_pos = False
                stop = False
            elif(short and ((data['close'] < entry_price) or entry_price==0)): #only add to position if original position is in profit!
                entry_price =  data['close']
                if(stopType == "atr"):
                    stopPrice = entry_price + data['atr']*50
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
        visual_data.loc[idx] = [data['timestamp'], capital]

    backtestfile = Path("BacktestData",currentTime + "_ATR" + str(atrperiod) + "_KP" + str(kperiod) + "_KSMA" + str(ksma) + ".txt")
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
    visual_data.to_csv('Plotting//' + currentTime + '.csv')

    visualize_trades(visual_data, currentTime)

    return [signal_params, capital]

def visualize_trades(df, currentTime):
    import matplotlib.pyplot 
    from matplotlib import pyplot as plt

    list_of_datetimes = df['timestamp'].tolist()
    list_of_datetimes = [t[:-6] for t in list_of_datetimes]
    l = [datetime.strptime(t, "%Y-%m-%d %H:%M:%S") for t in list_of_datetimes]
    values = df['capital'].tolist()
    dates = matplotlib.dates.date2num(l)
    matplotlib.pyplot.plot_date(dates, values,'-b')
    plt.xticks(rotation=90)
    plt.savefig('Plotting//'+ currentTime + '.png')

#ATR
atrperiod_v = [5,50]
#KELTNER
kperiod_v = [15,30]
ksma_v = [True]
keltner_v = [True]
#ENGULFING CANDLES
engulf_v = [True]
engulfthreshold_v = [.5, .75 , 1]
ignoredoji_v = [True,False]
#TRADE TYPES
trade_v = ['dynamic', 'long']
#POSITION SIZES
posmult_v = [2, 4, 8]

a = [atrperiod_v, kperiod_v, ksma_v, keltner_v, engulf_v, ignoredoji_v, trade_v, posmult_v, engulfthreshold_v]
combinations = list(itertools.product(*a))

def get_all_combinations(params):
    params_to_try = [{'atrperiod':l[0], 'kperiod':l[1], 'ksma':l[2], 'keltner':l[3] , 'engulf':l[4], 'ignoredoji':l[5], 'trade':l[6],  'posmult':l[7], 'engulfthreshold': l[8]} for l in combinations]
    return params_to_try

#params_to_try = [{'keltner': True, 'engulf': True, 'kperiod': 30, 'ksma': True, 'atrperiod': 5, 'ignoredoji': True, 'engulfthreshold': 1, 'trade': 'dynamic', 'posmult': 32}]

def genIndicators(candleamount, keltner_params, engulf_params, atrperiod_v):
    #Generate set of unique Keltner values
    kelt_df = pd.DataFrame(keltner_params)
    kelt_pairs = set()
    for kpreriod in keltner_params[0]:
        for ksma in keltner_params[1]:
            kelt_pairs.add((kpreriod, ksma))

    #Generate set of unqiue engulfing signals
    engulf_df = pd.DataFrame(engulf_params)
    engulf_pairs = set()
    for engfulfthreshold in engulf_params[0]:
        for ignoredoji in engulf_params[1]:
            engulf_pairs.add((engfulfthreshold, ignoredoji))

    keltner_pairs = list(kelt_pairs)
    engulf_pairs = list(engulf_pairs)
    atr_pairs = list(set(atrperiod_v))

    getSignals.saveKeltnerBands(candleamount, params=keltner_pairs)
    getSignals.saveKeltnerSignals(candleamount, params=keltner_pairs)
    getSignals.saveEngulfingSignals(candleamount, params=engulf_pairs)
    getSignals.saveATR(candleamount, params=atr_pairs)

def saveIndicators(combinations, candleamount=candleamount):
    atrperiod_v = [l[0] for l in combinations]
    kperiod_v = [l[1] for l in combinations]
    ksma_v = [l[2] for l in combinations]
    keltner_params = [kperiod_v, ksma_v]
    engulf_params = [engulfthreshold_v, ignoredoji_v]
    print('got here')
    genIndicators(candleamount, keltner_params, engulf_params, atrperiod_v)

#example of generating all indicators for defined params for a specific length of time
    #they go into Indicators/ folder, saved a csv by their parameters
        #ATRs = p<period>.csv
        #Keltners = kp<kperiod>_sma<ksma=True|False>.csv
print("got here")
saveIndicators(combinations, candleamount=candleamount)

###create multithread pool w/ number of threads being number of combinations###
#print("thread amount:", len(params_to_try))
#pool = ThreadPool(len(params_to_try))
#pool.uimap(lambda signal_params, : saveIndicators(combinations=combinations), params_to_try)
#results = pool.u  imap(lambda signal_params, : backtest_strategy(signal_params=signal_params), params_to_try)
#print("THE BEST SIGNALS ARE", max(list(results), key=lambda x:x[1]))

#backtest_strategy(candleamount, signal_params=params_to_try[0])
