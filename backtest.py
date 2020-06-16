import pandas as pd
import numpy as np
#from enum import Enum
import itertools
import time
from datetime import datetime
from pathos.threading import ThreadPool
from multiprocessing import Pool, Queue #, Process, Condition
from threading import current_thread
#import os
#import tracemalloc
#from functools import partial
import matplotlib.pyplot
from matplotlib import pyplot as plt
from datetime import datetime
import getSignals as getSignals
from indicators import Signal
from statistics import mean
import math

#Set data for backtest
candleamount = 1440*100
  #Remove this later in favor of running several symbols (pairs) at once
symbol = 'XBTUSD'
ctime = "1m"
#Greater percision speeds up the multithreading algorithm
##It does not reduce accuracy, but if the number is too high, the backtest will say so
percision = 24
visualize=False
capital = 1000
'''
######## PARAMETERS TO RUN BACKTEST ON ########
#ATR
atrperiod_v = [5]
#KELTNER
kperiod_v = [15, 30, 50]
ksma_v = [True, False]
keltner_v = [True]
#ENGULFING CANDLES
engulf_v = [True]
engulfthreshold_v = [1]
ignoredoji_v = [True]
#TRADE TYPES
trade_v = ['dynamic']
#POSITION SIZES
posmult_v = [2, 4, 8]
stoptype_v = ['atr']
symbol_v = ['XBTUSD']
params = [atrperiod_v, kperiod_v, ksma_v, keltner_v, engulf_v, ignoredoji_v, trade_v, posmult_v, engulfthreshold_v, stoptype_v, symbol_v]
################################################
'''
######## PARAMETERS TO RUN BACKTEST ON ########
#ATR
atrperiod_v = [15]
stopmult_v = [15]
tmult_v = [.0125]
#KELTNER
kperiod_v = [20]
ksma_v = [True]
keltner_v = [True]
#ENGULFING CANDLES
engulf_v = [True]
engulfthreshold_v = [.75]
ignoredoji_v = [True]
#TRADE TYPES
trade_v = ['dynamic']
#POSITION SIZES
posmult_v = [8]
stoptype_v = ['atr']
symbol_v = ['XBTUSD']
params = [atrperiod_v, kperiod_v, ksma_v, keltner_v, engulf_v, ignoredoji_v, trade_v, posmult_v, engulfthreshold_v, stoptype_v, stopmult_v, tmult_v, symbol_v]
################################################


combinations = list(itertools.product(*params))
params_to_try = [{'atrperiod':l[0], 'kperiod':l[1], 'ksma':l[2], 'keltner':l[3] , 'engulf':l[4], 'ignoredoji':l[5], 'trade':l[6],  'posmult':l[7], 'engulfthreshold': l[8], 'stoptype': l[9], 'stopmult': l[10], 'tmult': l[11], 'symbol': l[12]} for l in combinations]
#params_to_try = [{'keltner': True, 'engulf': True, 'kperiod': 30, 'ksma': True, 'atrperiod': 5, 'ignoredoji': True, 'engulfthreshold': 1, 'trade': 'dynamic', 'posmult': 32}]

#example of generating all indicators for defined params for a **ADD THIS LATER: specific length of time***
    #they go into Indicators/<XBTUSD|ETHUSD> folder, saved a csv by their parameters
        #ATRs = p<period>.csv
        #Keltners = kp<kperiod>_sma<ksma=True|False>.csv

#multiprocessing condition
#check = Condition()

def generateTargetPrice(entry_price, trade, tmult):
    if(trade=='short'):
        return entry_price - entry_price * tmult
    if(trade=='long'):
        return entry_price + entry_price * tmult

def backtest_strategy(candleamount, capital, signal_params, candles, safe): #trade= long, short, dynamic
    symbol = signal_params['symbol']
    kperiod = signal_params['kperiod']
    ksma = signal_params['ksma']
    atrperiod = signal_params['atrperiod']
    trade = signal_params['trade']
    ignoredoji = signal_params['ignoredoji']
    engulfthreshold = signal_params['engulfthreshold']
    posmult = signal_params['posmult']
    stopType= signal_params['stoptype']
    stopmult = signal_params['stopmult']
    tmult = signal_params['tmult']

    candle_data = candles
    #replace later with a less memory-heavy solution for finding candle indices without index reset
    old_candle_data = candle_data

    candle_data = candle_data.reset_index(drop=True)
    capital_data = pd.DataFrame(columns= ['timestamp', 'capital'])

    entry_price = 0
    profit = 0
    position_amount = 0
    static_position_amount = capital * .1
    fee = position_amount * 0.00075
    have_pos = False
    stop_loss = .1
    stop = False
    stopPrice=0
    #currentTime = datetime.now().strftime("%Y%m%d-%H%M")
    short_b = False
    long_b = False
    opposite_b = False
    target_price = 0
    targetHit = False
    lastidx = 0
    #print(candle_data)
    #time.sleep(1000)
    for idx, data in candle_data.iterrows():
        #print(safe)
        #print(data['timestamp'])
        #for safe point debugging
        #if(safe == False and ((data[-signal_len:]['S'] != "Signal.SELL") and (data[-signal_len:]['S'] != "Signal.BUY"))):
        #    print(idx)
        #    print("NONE")
        #    return([True, 1000000000])
        #print("got here")
        currentpoint = list(old_candle_data.index)[idx]
        #print(currentpoint, current_thread().name)

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
            if(((position_amount > 0) and (data['close'] < stopPrice)) or (((position_amount < 0)) and (data['close'] > stopPrice))):
                stop = True
                #print("!!!!!! STOP PRICE HIT !!!!!!", idx, position_amount, have_pos, data['timestamp'], current_thread().name)
                #print("Price:", data['close'])
                #print("ATR stop threshold: ", data['atr']*stopmult)
                #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                if(safe):
                    return(currentpoint)
        ##For multithreading algorithm debugging
        #if(data['timestamp'] == pd.to_datetime('2020-05-29 22:54:00+00:00')):
        #    print(data['S'], entry_price, targetHit, stop, idx, data['timestamp'], current_thread().name)
        if(data['S'] == "Signal.BUY") or (stop and position_amount < 0) or (data['close'] < target_price and position_amount < 0):
            if(data['S'] == "Signal.BUY"):
                long_b = True
                if(short_b):
                    opposite_b = True
                short_b = False
            if((short and have_pos) and (position_amount < 0)):
            #short is only a constant parameter to determine if we are shorting at all
            #position_amount determines if we are currently short or long
                profit = position_amount * ((data['close'] - entry_price)/entry_price)
                capital += profit
                capital -= fee
                #print("######## SHORT EXIT ########", capital, idx, data['timestamp'], current_thread().name)
                if (data['close'] < target_price and position_amount < 0):
                    #print("!!! TARGET PRICE REACHED !!!")
                    targetHit = True
                position_amount = 0
                #print("Exit price:", data['close'])
                #print("Turnover:", profit - fee*2)
                #print("Stop, thread: ", stop)
                #print("############################")
                entry_price = 0
                have_pos = False
                stop = False
            if(long and (data['close'] > entry_price or entry_price==0) and (targetHit == False)):
                entry_price = data['close']
                if(stopType == "atr"): #and if we already have a position, our stop moves up to reflect second entry
                    stopPrice = entry_price - data['atr']*stopmult
                elif(stopType == "perc"):
                    stopPrice = entry_price * (1-stop_loss)
                if(have_pos == False):
                    target_price = generateTargetPrice(entry_price, 'long', tmult)
                    position_amount = static_position_amount
                else:
                    position_amount += posmult*static_position_amount #we only get up to this point if our position is positive
                fee = position_amount*0.00075
                capital -= fee
                #print("######## LONG ENTRY ########", capital, idx, data['timestamp'], current_thread().name)
                #print("Entry price:", entry_price)
                #print("Target price:", target_price)
                #print("Stop loss:", stopPrice)
                #print("Current position:", position_amount)
                #print("Thread: ", current_thread().name)
                #print("############################")
                have_pos = True
                stop = False
            targetHit = False
        elif(data['S'] == "Signal.SELL") or (stop and position_amount > 0) or (data['close'] > target_price and position_amount > 0):
            if(data['S'] == "Signal.SELL"):
                short_b = True
                if(long_b):
                    opposite_b = True
                long_b = False
            if((long and have_pos) and (position_amount > 0)):
                profit = position_amount * ((data['close'] - entry_price)/entry_price)
                capital += profit
                capital -= fee
                #print("######### LONG EXIT ########", capital, idx, data['timestamp'], current_thread().name)
                if (data['close'] > target_price and position_amount > 0):
                    #print("!!! TARGET PRICE REACHED !!!")
                    targetHit = True
                position_amount = 0
                #print("Exit price:", data['close'])
                #print("Turnover:", profit - fee*2)
                #print("Stop, thread: ", stop, current_thread().name)
                #print("############################")
                entry_price = 0
                have_pos = False
                stop = False
            if(short and (data['close'] < entry_price or entry_price==0) and (targetHit == False)): #only add to position if original position is in profit!
                entry_price =  data['close']
                if(stopType == "atr"):
                    stopPrice = entry_price + data['atr']*stopmult
                elif(stopType == "perc"):
                    stopPrice = entry_price * (1+stop_loss)
                if(have_pos == False):
                    target_price = generateTargetPrice(entry_price, 'short', tmult)
                    position_amount = -1*static_position_amount
                else:
                    position_amount -= posmult*static_position_amount #we only get up to this point if our position is negative
                #print("####### SHORT ENTRY ########", capital, idx, data['timestamp'], current_thread().name)
                #print("Entry price:", entry_price)
                #print("Target Price:", target_price)
                #print("Stop loss:", stopPrice)
                #print("Current position:", position_amount)
                #print("Thread: ", current_thread().name)
                #print("############################")
                have_pos = True
                fee = abs(position_amount*0.00075)
                capital -= fee
                stop = False
            targetHit = False
        if(safe):
            if opposite_b: return(currentpoint)
            #cRange = candle_data.index[-1] - candle_data.index[0] 
            if ((idx > candleamount/2) or (idx == candle_data.index[-1])):
            #if (idx > candle_data.index[-1]):
                #print("idx: ", idx)
                #print("candleamount: ", candleamount)
                #print("candle index: ", candle_data.index[-1])
                return(-1)
        lastidx = idx
        
        capital_data.loc[idx] = [data['timestamp'], capital]
    #capital_data.to_csv('Plotting//' + symbol + '//' + currentTime + '.csv')

    #if visualize:
    #    visualize_trades(capital_data)

    print("Backtest for given param completed, and results were saved to Backtest/" + symbol, lastidx, current_thread().name)
    timestamp = capital_data.iloc[-1]['timestamp']
    return(timestamp, capital_data)

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

def saveIndicators(combinations=combinations, candleamount=candleamount):
    atrperiod_v = [l[0] for l in combinations]
    kperiod_v = [l[1] for l in combinations]
    ksma_v = [l[2] for l in combinations]
    keltner_params = [kperiod_v, ksma_v]
    engulf_params = [engulfthreshold_v, ignoredoji_v]
    genIndicators(candleamount, keltner_params, engulf_params, atrperiod_v)
    print("Indicators generated, and results were saved to Indicators/" + symbol)
    return("indprocess done")

def visualize_trades(df, backtestDir):
    list_of_datetimes = df['timestamp'].tolist()
    list_of_datetimes = [str(t)[:-6] for t in list_of_datetimes]
    l = [datetime.strptime(t, "%Y-%m-%d %H:%M:%S") for t in list_of_datetimes]
    values = df['capital'].tolist()
    dates = matplotlib.dates.date2num(l)
    matplotlib.pyplot.plot_date(dates, values,'-b')
    plt.xticks(rotation=90)
    plt.savefig('Plotting//'+ backtestDir + 'a' + '.png')
    print("Visualization generated, and saved to Plotting/" + symbol)
    return("visprocess done")

#update later to allow backtesting different pairs simultaneously
candleData = pd.read_csv(symbol_v[0] + "-" + ctime + "-data.csv", sep=',').drop(columns=['lastSize','turnover','homeNotional','foreignNotional'])
xbtusd_su = 1111
ethusd_su = 1111
find_su = False #debugging feature for backtest optimization
def backtest_mt(params):
    global capital
    su = None
    saveIndicators(candleamount=candleamount)
    #fix later
    candleSplice = candleData.tail(candleamount)

    atrseries = pd.Series(dtype=np.uint16)
    keltner_signals = pd.Series(dtype=object)
    engulf_signals = pd.Series(dtype=object)
    signals = pd.DataFrame(columns=['S'])
    atrperiod = params['atrperiod']
    #candleSplice = candleSplice.reset_index(drop=True)

    if (params['keltner'] == True) and (params['engulf'] == True):
        engulf_signals = pd.read_csv('IndicatorData//' + params['symbol'] + '//Engulfing//' + "SIGNALS_t" + str(params['engulfthreshold']) + '_ignoredoji' + str(params['ignoredoji']) + '.csv', sep=',')
        keltner_signals = pd.read_csv('IndicatorData//' + params['symbol'] + '//Keltner//' + "SIGNALS_kp" + str(params['kperiod']) + '_sma' + str(params['ksma']) + '.csv', sep=',')
        signals = pd.concat([engulf_signals, keltner_signals], axis=1)
        signals.columns = ["E", "K"]    
        signals['S'] = np.where((signals['E'] == signals['K']), Signal(0), signals['E'])
    elif(params['keltner'] == True):
        keltner_signals = pd.read_csv('IndicatorData//' + params['symbol'] + '//Keltner//' + "SIGNALS_kp" + str(params['kperiod']) + '_sma' + str(params['ksma']) + '.csv', sep=',')
        signals['S'] = np.array(keltner_signals).reshape(1, len(keltner_signals))[0]
    elif(params['engulf'] == True):
        engulf_signals = pd.read_csv('IndicatorData//' + params['symbol'] + '//Engulfing//' + "SIGNALS_t" + str(params['engulfthreshold']) + '_ignoredoji' + str(params['ignoredoji']) + '.csv', sep=',')
        signals['S'] = np.array(engulf_signals).reshape(1, len(engulf_signals))[0]
    print(signals['S'])
    #signals.to_csv('BacktestData//Signals//' + currentTime + '.csv')
    atrseries = pd.read_csv('IndicatorData//' + params['symbol'] + "//ATR//" + "p" + str(atrperiod) + '.csv', sep=',')
    copyIndex = candleSplice.index
    candleSplice = candleSplice.reset_index(drop=True)
    #candleSplice.merge(atrseries, left_index=True)
    #candleSplice.merge(signals['S'], right_on='S', left_index=True)
    candleSplice = pd.DataFrame.join(candleSplice, atrseries)
    candleSplice = pd.DataFrame.join(candleSplice, signals['S'])   #COMBINE SIGNALS AND CANDLE DATA
    candleSplice.index = copyIndex
    candleSplice['timestamp'] = pd.to_datetime(candleSplice.timestamp)
    finalCapitalData = None
    currentTime = datetime.now().strftime("%Y%m%d-%H%M")
    backtestDir = params['symbol'] + '//' + "len" + str(candleamount) + "_k" + str(params['keltner']) + "_e" + str(params['engulf']) + "_id" + str(params['ignoredoji']) + "_eThrs" + str(params['engulfthreshold']) + "_ATR" + str(params['atrperiod']) + "_kP" + str(params['kperiod']) + "_kSMA" + str(params['ksma']) + "_pm" + str(params['posmult']) + "_ST" + params['stoptype'] + "_sm" + str(params['stopmult']) + "_tm" + str(params['tmult']) + "_TR" + params['trade']

    bt_profit = 0

    if(percision != 1):
        isafe = []
        candleSplit = []
        initialLength = len(candleSplice)
        firstStart = candleSplice.index[0]
        lastDistanceSafe = None
        if params['symbol'] == 'XBTUSD':
            su = xbtusd_su
        elif params['symbol'] == 'ETHUSD':
            su = ethusd_su
        for i in range(percision-1):
            #abs() is a temporary fix to running the backtest on short intervals
            isafe.append((i+1)*((abs(initialLength-percision*su))/percision)+i*su)
        #candleSplit = list(np.array_split(candleSplice, percision))
        #candleSplit = list(candleSplit)
        for i in isafe:
            ia = int(i)
            if isafe.index(i) != 0:
                candleSplit.append(candleSplice.iloc[int(isafe[isafe.index(i)-1]):ia+1])
            lastDistanceSafe = ia
                #print("lds", lastDistanceSafe)
           # else:
                #candleSplit.append(candleSplice.iloc[:ia+1])
                #print("lds", lastDistanceSafe)
        #if(len(isafe) > 1):
        candleSplit.append(candleSplice.iloc[lastDistanceSafe:])

        #print(candleSplit)
        #time.sleep(100)
        #generate parameters for multithreading
        safe_length = len(candleSplit)
        safe_candleamount = np.repeat(candleamount, safe_length).tolist()
        safe_capital = np.repeat(capital, safe_length).tolist()
        safe_params = np.repeat(params, safe_length).tolist()

        withSafe = np.repeat(True, safe_length).tolist()

        print("safe thread amount:", safe_length)
        #create multithread pool
        start = time.time()
        #print(candleSplit)
        #time.sleep(1000)
        pool = ThreadPool(safe_length)
        
        #run initial chunks multithreaded to find safepoints
        safe_results = pool.uimap(backtest_strategy, safe_candleamount, safe_capital, safe_params, candleSplit, withSafe)
        
        pool.close()    #Compute anything we need to while threads are running
        candleSafe = []
        final_length = safe_length + 2
        withoutSafe = np.repeat(False, final_length).tolist()
        final_candleamount = np.repeat(candleamount, final_length).tolist()
        final_capital = np.repeat(capital, final_length).tolist()
        final_params = np.repeat(params, final_length).tolist()
        static_capital = capital

        safePoints = list(safe_results) ######################################
        #time.sleep(1000)
        pool.join()

        for i in safePoints:
            if i == -1:
                backtest_mt.q.put('Not all safe points found for given percision. Reduce percision, or increase timeframe')
                return
        safePoints = sorted(safePoints)

        if find_su:
            su = []
            for i, point in enumerate(safePoints):
                su.append(point - candleSplit[i].index[0])
            suAvg = mean(su)
                #only works on evenly spliced chunks
            chunkLength = len(candleSplit[0])
            backtest_mt.q.put(["su average:", suAvg, ' / ', chunkLength])
            return(su)

        print("safe points:", safePoints)
        idx = 0
        for i in safePoints:
            ia = i - firstStart
            idx = safePoints.index(i)
            if safePoints.index(i) != 0:
                candleSafe.append(candleSplice.iloc[lastDistanceSafe-idx:ia+1])
                lastDistanceSafe = ia + 1
            else:
                candleSafe.append(candleSplice.iloc[:ia+1])
                lastDistanceSafe = ia + 1
        candleSafe.append(candleSplice.iloc[lastDistanceSafe-idx:])

        print("final thread amount:", final_length)
        #print(candleSafe)
        #time.sleep(10000)
        fpool = ThreadPool(final_length)
        final_results = fpool.uimap(backtest_strategy, final_candleamount, final_capital, final_params, candleSafe, withoutSafe)
        fpool.close()
        final_result = list(final_results)
        fpool.join()

        ordered_result = sorted(final_result, key=lambda x: x[0])
        for i in range(len(ordered_result)):
            #print(final_result.index)
            if i != 0:
                #for non-static position size:
                ##capital += capital*((i[1]-static_capital)/static_capital)
                ordered_result[i][1]['capital'] += bt_profit
                bt_profit = ordered_result[i][1].iloc[-1]['capital']-static_capital
                finalCapitalData = pd.concat([finalCapitalData, ordered_result[i][1]], ignore_index=True)
            else:
                bt_profit = ordered_result[i][1].iloc[-1]['capital']-static_capital
                finalCapitalData = pd.DataFrame(ordered_result[i][1])
        capital = finalCapitalData['capital'].iloc[-1]
    else:
        #run chunks spliced by safepoints multithreaded to retrieve fully accurate results
        final_results = backtest_strategy(candleamount, capital, params, candleSplice, False)
        final_result = list(final_results)
        capital = str(final_result[1]['capital'].iloc[-1])
        finalCapitalData = final_result[1]

    print(finalCapitalData)
    #time.sleep(1000)
    visualize_trades(finalCapitalData, backtestDir)
    saveBacktest(capital, params, backtestDir)
    backtest_mt.q.put(capital)
    end = time.time()
    print("Thread time: ", end-start)
    return('done')

def saveBacktest(capital, params, backtestDir):
    f = open('BacktestData//' + backtestDir + '.txt', 'a')
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
    f.write("Keltner:" + str(params['keltner']) + ", Engulf:" + str(params['engulf']))
    f.write("\nPosition multiplier: ")
    f.write(str(params['posmult']))
    f.write("\nATR Period: ")
    f.write(str(params['atrperiod']))
    f.write("\nKeltner Period: ")
    f.write(str(params['kperiod']))
    f.write("\nKeltner SMA (EMA if false): ")
    f.write(str(params['ksma']))
    f.write("\nIgnore Doji: ")
    f.write(str(params['ignoredoji']))
    f.write("\nEngulfing Threshold: ")
    f.write(str(params['engulfthreshold']))
    f.write("\nTrade Type: ")
    f.write(params['trade'])
    f.write('\n---------------------------\n')
    f.close()

#Monkey patch for multiprocessing queues (for messages, like results)
def f_init(q):
    backtest_mt.q = q

if __name__ == '__main__': 
    q = Queue()
    pLen = len(params_to_try)
    with Pool(pLen, f_init, [q]) as pool:
        print("Running backtest for", pLen, "strategies with multiprocessing...")
        start = time.time()
        res = pool.imap_unordered(backtest_mt, params_to_try)
        pool.close()
        pool.join()
        end = time.time()
        print("Backtest time: ", end-start)
        print("Backtest completed for all given params, and all generated data was saved :)")
        for i in range(len(params_to_try)):
            print("queue:", q.get())
        #check.release()
        #capital_data = list(zip(*result))[0]

#### MULTIPROCESSING DOES NOT RETURN CODE ERRORS. USE THIS FOR DEBUGGING ####
#print(len(params_to_try))
#for i in params_to_try:
#    backtest_mt(i)
#print("THE BEST SIGNALS ARE:", max(param_data, key=lambda x:x[1]))
#############################################################################
