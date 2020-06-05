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
    candle_data = pd.DataFrame(columns=['open', 'close','high', 'low', 'candle size', 'type'])
    # iterate over rows with iterrows()
    for index, data in candles.head(candleamount).iterrows():
        #determine if candles are of opposite type
        if data['open'] < data['close']:
            type = "green"
        elif data['open'] > data['close']:
            type = "red"
        else:
            type = "abs_doji"
     #append size
        candle_data.loc[index] = [data['open'], data['close'],data['high'], data['low'], abs(data['open']-data['close']), type]
    return candle_data

#realtime func
def engulfingsignals(curr_row, prev_row, threshold = 1, ignoredoji = False):
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

def atrseries(candles, period=10, fillna=True):
    atr = ta.volatility.AverageTrueRange(candles["high"], candles["low"], candles["close"], n=period, fillna=fillna)
    series = pd.Series()
    series = atr.average_true_range()
    return(series)

#back-test on the series extrapolated from price data
def get_engulf_signals(candles, candleamount = MIN_IN_DAY, threshold=1, ignoredoji=False):
    #first generate a candle-series!
    candles = candle_df(candles, candleamount)
    signals=[Signal.WAIT]
    #generate a trade signal for every candle except for the last, and store in the list we created
    prev_row = candles.iloc[0]
    for i,row in candles.head(candleamount).iloc[1:].iterrows():
        signals.append(engulfingsignals(row, prev_row, threshold, ignoredoji))
        prev_row = row
    return signals

def get_keltner_signals(candles, candleamount = MIN_IN_DAY, ma=10, threshold = 1, ignoredoji = False, sma=True):
    indicator_kelt = ta.volatility.KeltnerChannel(high=candles["high"], low=candles["low"], close=candles["close"], n=ma, fillna=True, ov=sma)
    kseries = pd.DataFrame(columns=['hband', 'lband'])
    # Add Bollinger Bands features
    kseries["hband"] = indicator_kelt.keltner_channel_hband_indicator()
    kseries["lband"] = indicator_kelt.keltner_channel_lband_indicator()
    signals=[]

    for i,row in kseries.head(candleamount).iterrows():
        signals.append(keltnersignals(row))
    return signals
    '''
def backtest_strat(candles, candleamount = MIN_IN_DAY, signal_params = {'keltner': True, 'engulf':True,  'kperiod':40, 'ksma':True, 'atrperiod':30, 'capital': 1000, 'ignoredoji':False, 'engulfthreshold': 1, 'trade':"dynamic"}, capital = 100):
    
    signals = pd.DataFrame()
    if(signal_params['keltner'] == True):
       signals['keltner'] = get_keltner_signals(candles, candleamount= candleamount)
       #print("KELTNER SIGNALS", signals.groupby('keltner'))

    if(signal_params['engulf'] == True):
        signals['engulf'] = get_engulf_signals(candles, candleamount=candleamount, engulfthreshold, ignoredoji)

    position = 0
    capital = 1000
    buy_price = 0
    profit = 0
    signals.to_csv('signals')
    position_size = 0.05
    position_amount = capital * position_size
    have_pos = False
    for idx, data in signals.head(candleamount).iterrows():
        if(all([v == Signal.BUY for v in data]) and have_pos == False):
            buy_price =  candles.loc[idx,'close']
            position = 1
            capital -= position_amount*0#fee
        elif(all([v == Signal.SELL for v in data]) and have_pos == True):
            profit = position_amount * ((candles.loc[idx+1,'open'] - buy_price)/buy_price)
            #100 * (200-400/400) = 100
            capital += profit
            capital -= position_amount*0  # fee
            position = 0
            have_pos = False
    print('profit', capital)
    #for i in candles.head(candleamount).iterrows():
    return signals'''
#def backtest_strat(candles, candleamount = MIN_IN_DAY, signals_to_use = {'keltner': True, 'engulf':True,  'kperiod':40, 'ksma':True, 'atrperiod':30, 'capital': 1000, 'ignoredoji':False, 'engulfthreshold': 1, 'trade':"dynamic"}, capital = 100):

def backtest_strategy(candleamount = 1440, capital = 1000, signal_params = {'keltner': True, 'engulf':True,  'kperiod':40, 'ksma':True, 'atrperiod':30, 'ignoredoji':False, 'engulfthreshold': 1, 'trade':"dynamic"}): #trade= long, short, dynamic
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
                    position_amount -= posmult*static_position_amount #we only get up to this point if our position is negative
                print("####### SHORT ENTRY ########")
                print("Entry price:", entry_price)
                print("Stop loss:", stopPrice)
                print("Current position:", position_amount)
                print("############################")
                have_pos = True
                fee = abs(position_amount*0.00075)
                capital -= fee

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

    return [signal_params, profit]
    #for i in candles.head(candleamount).iterrows():


#define data to test with
candles = pd.read_csv("XBTUSD-1m-data.csv", sep=',')

#create multithread pool
pool = ThreadPool()

from functools import partial
#mapfunc = partial(backtest_strategy, 'atrperiod')

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

atrperiod_v = [5,10,20,30]
kperiod_v = [10,20,30,40]
ksma_v = [True]
keltner_v = [True,False]
engulf_v = [True,False]
ignoredoji_v = [True,False]
trade_v = ['dynamic', 'long']
posmult_v = [ 2, 4, 8]
engulfthreshold_v = [.75 , 1]

a = [atrperiod_v, kperiod_v, ksma_v, keltner_v, engulf_v, ignoredoji_v, trade_v, posmult_v, engulfthreshold_v]
combinations = list(itertools.product(*a))
atrperiod_v = [l[0] for l in combinations]

atrperiod_v = []
for l in combinations:
    atrperiod_v.append(l*3)

kperiod_v = [l[1] for l in combinations]
ksma_v = [l[2] for l in combinations]
params = [ {'keltner':l[3] , 'engulf':l[4],  'kperiod':l[1], 'ksma':l[2], 'atrperiod':l[0], 'ignoredoji':l[5], 'engulfthreshold': l[8], 'trade':l[6], 'posmult':l[7]} for l in combinations]
#{'keltner': True, 'engulf':True,  'kperiod':40, 'ksma':True, 'atrperiod':30, 'ignoredoji':False, 'engulfthreshold': 1, 'trade':"dynamic"}
print("Params", params)
#atrperiod_v = [5,10,20,30]
#kperiod_v = [10,20,30,40]
#ksma_v = [True, True, True, True]
#results = pool.uimap(lambda atrperiod, kperiod, ksma, : backtest_strategy(atrperiod=atrperiod, kperiod=kperiod, ksma=ksma), atrperiod_v, kperiod_v, ksma_v)
#backtest_strategy()
results = pool.uimap(lambda signal_params, : backtest_strategy(signal_params=signal_params), params)

print("THE BEST SIGNALS ARE", max(list(results), key=lambda x:x[1])

    #terminate process on key press
    #stop_char=""
    #while stop_char.lower() != "q":
    #    stop_char=input("Enter 'q' to quit ")
    #print("terminate process")
    #p.terminate()