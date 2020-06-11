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
import dill
import itertools
from functools import partial
from datetime import datetime
from indicators import Signal, candle_df
import getSignals as getSignals
import matplotlib.pyplot
from multiprocessing import Process, Pool, Condition, Queue
from matplotlib import pyplot as plt
import threading
import os
import pickle

#Set data for backtest
symbol = "XBTUSD"
candleamount = 14400
ctime = "1m"
visualize=False
candleData = pd.read_csv(symbol + "-" + ctime + "-data.csv", sep=',').drop(columns=['lastSize','turnover','homeNotional','foreignNotional'])

def generateTargetPrice(entry_price, trade):
    if(trade=='short'):
        return entry_price - entry_price *.02
    if(trade=='long'):
        return entry_price + entry_price  *.02


def backtest_strategy(candleamount, capital, signal_params, candles, safe, symbol=symbol): #trade= long, short, dynamic
    atrseries = pd.Series(dtype=np.uint16)
    signals = pd.DataFrame()
    keltner_signals = pd.Series(dtype=object)
    engulf_signals = pd.Series(dtype=object)
    #print(signal_params)
    kperiod = signal_params['kperiod']
    ksma = signal_params['ksma']
    atrperiod = signal_params['atrperiod']
    trade = signal_params['trade']
    ignoredoji = signal_params['ignoredoji']
    engulfthreshold = signal_params['engulfthreshold']
    posmult = signal_params['posmult']
    candle_data = candles

    #start and exit points for multithreading
    #startpoint = list(candle_data.index)[0]
    #exitpoint = exitpoints
    #print("startpoint: ", startpoint)
    #print("exitpoint: ", exitpoint)
    #print("range: ", ((exitpoint-startpoint)/2))

    if (signal_params['keltner'] == True) and (signal_params['engulf'] == True):
        engulf_signals = pd.read_csv('IndicatorData//' + symbol + '//Engulfing//' + "SIGNALS_t" + str(signal_params['engulfthreshold']) + '_ignoredoji' + str(signal_params['ignoredoji']) + '.csv', sep=',')
        keltner_signals = pd.read_csv('IndicatorData//' + symbol + '//Keltner//' + "SIGNALS_kp" + str(signal_params['kperiod']) + '_sma' + str(signal_params['ksma']) + '.csv', sep=',')
        signals = pd.concat([engulf_signals, keltner_signals], axis=1)
        signals.columns = ["E", "K"]    
        signals['S'] = np.where((signals['E'] == signals['K']), Signal(0), signals['E'])
    elif(signal_params['keltner'] == True):
        keltner_signals = pd.read_csv('IndicatorData//' + symbol + '//Keltner//' + "SIGNALS_kp" + str(signal_params['kperiod']) + '_sma' + str(signal_params['ksma']) + '.csv', sep=',')
        signals['S'] = keltner_signals
        #print("KELTNER SIGNALS", signals.groupby('keltner'))
    elif(signal_params['engulf'] == True):
        engulf_signals = pd.read_csv('IndicatorData//' + symbol + '//Engulfing//' + "SIGNALS_t" + str(signal_params['engulfthreshold']) + '_ignoredoji' + str(signal_params['ignoredoji']) + '.csv', sep=',')
        signals['S'] = engulf_signals
    atrseries = pd.read_csv('IndicatorData//' + symbol + "//ATR//" + "p" + str(atrperiod) + '.csv', sep=',')

    signal_len = len(signals.loc[0])
    #remove later
    old_candle_data = candle_data
    candle_data = candle_data.reset_index(drop=True)
    candle_data = pd.DataFrame.join(candle_data, atrseries)
    candle_data = pd.DataFrame.join(candle_data, signals['S'])   #COMBINE SIGNALS AND CANDLE DATA
    #print("CANDLE DATA")
    #print(candle_data)
    
    capital_data = pd.DataFrame(columns= ['timestamp', 'capital'])
    entry_price = 0
    profit = 0
    #currentTime = datetime.now().strftime("%Y%m%d-%H%M")
    #signals.to_csv('BacktestData//Signals//' + currentTime + '.csv')
    position_size = .1
    position_amount = 0
    static_position_amount = capital * position_size
    fee = position_amount * 0.00075
    have_pos = False
    stop_loss = .1
    stop=False
    stopPrice=0
    stopType="atr"
    
    #threadData = pd.DataFrame()
    safe_b = False
    for idx, data in candle_data.iterrows():
        #multithreading
        currentpoint = list(old_candle_data.index)[idx]
        if(safe and safe_b):
            return(currentpoint)
        #print("current point: ", currentpoint)
        #print("thread lock:", threadloc)
        #We do not have ATR data at the start of back-test (unless we look further back, which will not improve our accuracy by much)
            #So, if we do not have ATR (w/ fillna it makes it 0), we set dummy data for the ATR
        if(data['atr']==0):
            data['atr']=1
        long=True
        short=True
        if(trade=="long"):
            long=True
            short=False
        elif(trade=="short"):
            long=False
            short=True
        if(entry_price != 0 and stopPrice != 0):
            if(((position_amount > 0) and (data['open'] < stopPrice)) or (((position_amount < 0)) and (data['open'] > stopPrice))):
                stop=True
                print("!!!!!! STOP PRICE HIT !!!!!!")
                print("Price:", data['open'])
                print("ATR stop threshold: ", data['atr']*50)
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        if(data[-signal_len:]['S'] == "Signal.BUY") or (stop and position_amount < 0):
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
                safe_b = True
            elif(data['close'] > entry_price or entry_price==0):
                entry_price = data['close']
                if(stopType == "atr"): #and if we already have a position, our stop moves up to reflect second entry
                    stopPrice = entry_price - data['atr']*50
                elif(stopType == "perc"):
                    stopPrice = entry_price * (1-stop_loss)
                if(have_pos == False):
                    position_amount = static_position_amount
                else:
                    position_amount += posmult*static_position_amount #we only get up to this point if our position is positive
                #target_price = generateTargetPrice(entry_price, 'long')
                fee = position_amount*0.00075
                capital -= fee
                print("######## LONG ENTRY ########")
                print("Entry price:", entry_price)
                #print("Target price:", target_price)
                print("Stop loss:", stopPrice)
                print("Current position:", position_amount)
                print("############################")
                have_pos = True

        elif(data[-signal_len:]['S'] == "Signal.SELL") or (stop and position_amount > 0):
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
                safe_b = True
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
                #target_price = generateTargetPrice(entry_price, 'short')
                print("####### SHORT ENTRY ########")
                print("Entry price:", entry_price)
                #print("Target Price:", target_price)
                print("Stop loss:", stopPrice)
                print("Current position:", position_amount)
                print("############################")
                have_pos = True
                fee = abs(position_amount*0.00075)
                capital -= fee
        #elif((currentpoint > (startpoint + ((exitpoint-startpoint)/2)) and threadloc) and have_pos==False):
        #    lastSafePos = currentpoint
        #    return([idx, capital_data])
        #elif((currentpoint > (startpoint + ((exitpoint-startpoint)/2)) and threadloc==False) and have_pos==False):
        #    return([idx, capital_data])
        capital_data.loc[idx] = [data['timestamp'], capital]

    currentTime = datetime.now().strftime("%Y%m%d-%H%M")
    backtestfile = 'BacktestData//' + symbol + '//' + currentTime + "_ATR" + str(atrperiod) + "_KP" + str(kperiod) + "_KSMA" + str(ksma) + ".txt"
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
    #capital_data.to_csv('Plotting//' + symbol + '//' + currentTime + '.csv')

    #if visualize:
    #    visualize_trades(capital_data)

    print("Backtest for given param completed, and results were saved to Backtest/" + symbol)
    print("CAPITAL: ", capital)
    return(capital)

def genIndicators(candleamount, keltner_params, engulf_params, atrperiod_v):
    print('genIndicators')
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

    getSignals.saveKeltnerBands(candleData, candleamount, params=keltner_pairs)
    getSignals.saveKeltnerSignals(candleData, candleamount, params=keltner_pairs)
    getSignals.saveEngulfingSignals(candleData, candleamount, params=list(engulf_pairs))
    getSignals.saveATR(candleData, candleamount, params=atr_pairs)

def saveIndicators(combinations, candleamount=candleamount):
    atrperiod_v = [l[0] for l in combinations]
    kperiod_v = [l[1] for l in combinations]
    ksma_v = [l[2] for l in combinations]
    keltner_params = [kperiod_v, ksma_v]
    engulf_params = [engulfthreshold_v, ignoredoji_v]
    genIndicators(candleamount, keltner_params, engulf_params, atrperiod_v)
    print("Indicators generated, and results were saved to Indicators/" + symbol)
    return("indprocess done")

def visualize_trades(df):
    list_of_datetimes = df['timestamp'].tolist()
    list_of_datetimes = [t[:-6] for t in list_of_datetimes]
    l = [datetime.strptime(t, "%Y-%m-%d %H:%M:%S") for t in list_of_datetimes]
    values = df['capital'].tolist()
    dates = matplotlib.dates.date2num(l)
    matplotlib.pyplot.plot_date(dates, values,'-b')
    plt.xticks(rotation=90)
    plt.savefig('Plotting//'+ symbol + '//' + str(time.time()) + "_" + current_thread().name + '.png')
    print("Visualization generated, and saved to Plotting/" + symbol)
    return("visprocess done")

######## PARAMETERS TO RUN BACKTEST ON ########
#ATR
atrperiod_v = [5, 50]
#KELTNER
kperiod_v = [15]
ksma_v = [True]
keltner_v = [True]
#ENGULFING CANDLES
engulf_v = [True]
engulfthreshold_v = [1]
ignoredoji_v = [True,False]
#TRADE TYPES
trade_v = ['dynamic']
#POSITION SIZES
posmult_v = [2, 4, 8]
params = [atrperiod_v, kperiod_v, ksma_v, keltner_v, engulf_v, ignoredoji_v, trade_v, posmult_v, engulfthreshold_v]
################################################

combinations = list(itertools.product(*params))
params_to_try = [{'atrperiod':l[0], 'kperiod':l[1], 'ksma':l[2], 'keltner':l[3] , 'engulf':l[4], 'ignoredoji':l[5], 'trade':l[6],  'posmult':l[7], 'engulfthreshold': l[8]} for l in combinations]
#params_to_try = [{'keltner': True, 'engulf': True, 'kperiod': 30, 'ksma': True, 'atrperiod': 5, 'ignoredoji': True, 'engulfthreshold': 1, 'trade': 'dynamic', 'posmult': 32}]

#example of generating all indicators for defined params for a specific length of time
    #they go into Indicators/<XBTUSD|ETHUSD> folder, saved a csv by their parameters
        #ATRs = p<period>.csv
        #Keltners = kp<kperiod>_sma<ksma=True|False>.csv

#multiprocessing condition
#check = Condition()

safevalues = []
def backtest_mt(params):
    #check.acquire()
    bt_capital = 1000
    saveIndicators(combinations, candleamount=candleamount)
    ###create multithread pool w/ number of threads being number of combinations###
    #pool.uimap(lambda signal_params, : saveIndicators(combinations=combinations), params_to_try)
    #results = pool.uimap(lambda signal_params, : backtest_strategy(candleamount, signal_params=signal_params), params_to_try)
    #result = list(results)

    percision = 2
    candleSplice = candleData.tail(candleamount)
    candleSplit = list(np.array_split(candleSplice, percision))

    #generate parameters for multithreading
    #exitpoints = []
    #for i in candleSplit:
    #    exitpoints.append(list(i.index)[-1])
    safe_candleamount = np.repeat(candleamount, len(candleSplit)).tolist()
    safe_capital = np.repeat(bt_capital, len(candleSplit)).tolist()
    safe_params = np.repeat(params, len(candleSplit)).tolist()
    #safe_threadloc = np.repeat(False, len(candleSplit)/2).tolist()
    #safe_F_tl = np.repeat(True, len(candleSplit)/2).tolist()
    #threadloc.extend(F_tl)

    withSafe = np.repeat(True, len(candleSplit)).tolist()
    withoutSafe = np.repeat(False, len(candleSplit)).tolist()

    print("thread amount:", percision)
    tpool = ThreadPool(percision)
    #backtest_strategy(candleamount, capital, params, candles=candleData)
    start = time.time()
    #print("ca", len(candleamount))
    #print("cap", len(capital))
    #print("params", len(params))
    #print("Candlesplit len:", len(candleSplit))
    #print("exitpoints", len(exitpoints))
    #print("threadloc", len(threadloc))
    #time.sleep(1000)
    safe_results = tpool.uimap(backtest_strategy, safe_candleamount, safe_capital, safe_params, candleSplit, withSafe)
    safe_result = list(safe_results)
    safePoints = list(np.asarray(sorted(safe_result))+1)
    print("safe points:", safePoints)
    firstStart = candleSplice.index[0]
    firstDistanceSafe = safePoints[0] - firstStart
    lastDistanceSafe = None
    candleSafe = []
    for i in safePoints:
        print(i)
        ia = i - firstStart
        if safePoints.index(i) != 0:
            print("ia", ia)
            print("lastDistanceSafe", lastDistanceSafe)
            candleSafe.append(candleSplice.iloc[lastDistanceSafe:ia])
            lastDistanceSafe = ia
        else:
            candleSafe.append(candleSplice.iloc[:firstDistanceSafe])
            lastDistanceSafe = firstDistanceSafe
    candleSafe.append(candleSplice.iloc[lastDistanceSafe:])
    #print("candleSafe:", candleSafe)

    final_candleamount = np.repeat(candleamount, len(candleSafe)).tolist()
    final_capital = np.repeat(bt_capital, len(candleSafe)).tolist()
    final_params = np.repeat(params, len(candleSafe)).tolist()
    #final_threadloc = np.repeat(False, len(candleSplit)/2).tolist()
    #final_F_tl = np.repeat(True, len(candleSplit)/2).tolist()
    #final_threadloc.extend(F_tl)

    #print((final_candleamount))
    #print((final_capital))
    #print((final_params))
    #print((candleSafe))
    final_results = tpool.uimap(backtest_strategy, final_candleamount, final_capital, final_params, candleSafe, withoutSafe)
    final_result = list(final_results)
    static_capital = bt_capital
    for i in final_result:
        print("capital: ", i)
        bt_capital += bt_capital*((i-static_capital)/static_capital)

    #check.wait()

    backtest_mt.q.put(final_result)
    end = time.time()
    print("Thread time: ", end-start)
    return(final_result)

def f_init(q):
    backtest_mt.q = q

if __name__ == '__main__': 
    q = Queue()
    with Pool(None, f_init, [q]) as pool:
        print("Running backtest for all given params with multiprocessing...")
        #capital_data = list(zip(*result))[0]
        start = time.time()
        res = pool.imap_unordered(backtest_mt, params_to_try)
        pool.close()
        pool.join()

        for i in range(len(params_to_try)):
            print("queue:", q.get())
        #check.release()
        end = time.time()
        print("Backtest time: ", end-start)
        print("Backtest completed for all given params, and all generated data was saved :)")


#print("THE BEST SIGNALS ARE:", max(param_data, key=lambda x:x[1]))