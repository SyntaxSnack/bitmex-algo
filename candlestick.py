import pandas as pd
import numpy as np
from collections import Counter
from enum import Enum
import ta
from ta.utils import dropna
import sys

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
    

def backtest_strategy(candles, candleamount = MIN_IN_DAY, signals_to_use = {}, capital = 100, trade="dynamic"): #trade= long, short, dynamic
    signals = pd.DataFrame()
    if(signals_to_use['keltner'] == True):
       signals['keltner'] = get_keltner_signals(candles, candleamount=candleamount, ma=40, sma=True)
       #print("KELTNER SIGNALS", signals.groupby('keltner'))

    if(signals_to_use['engulf'] == True):
        signals['engulf'] = get_engulf_signals(candles, candleamount=candleamount)

    capital = 1000
    entry_price = 0
    profit = 0
    signals.to_csv('signals')
    position_size = 1
    position_amount = capital * position_size
    static_position_amount = capital * position_size
    fee = position_amount * 0.00075
    have_pos = False
    stop_loss = .1
    stop=False
    atr=atrseries(candles, period=30)
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
                    position_amount += static_position_amount #we only get up to this point if our position is positive
                fee = position_amount*0.00075
                capital -= fee
                print("######## LONG ENTRY ########")
                print("Entry price:", entry_price)
                print("Stop loss:", stopPrice)
                print("Current position:", position_amount)
                print("############################")
                have_pos = True
                print(position_amount)
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
                    position_amount -= static_position_amount #we only get up to this point if our position is negative
                print("####### SHORT ENTRY ########")
                print("Entry price:", entry_price)
                print("Stop loss:", stopPrice)
                print("Current position:", position_amount)
                print("############################")
                have_pos = True
                fee = abs(position_amount*0.00075)
                capital -= fee

    print('\n---------------------------')
    print('---- BACKTEST COMPLETE ----')
    print("Backtest time:", candleamount/1440, "days")
    print("Final capital:", capital)
    print("Total profit:", capital-1000)
    print('---------------------------')
    #for i in candles.head(candleamount).iterrows():
    return signals

candles = pd.read_csv("ETHUSD-1m-data.csv", sep=',')
#print(Counter(get_engulf_signals(candles)))
#print(Counter(get_keltner_signals(candles)))
print(backtest_strategy(candles, candleamount = 14400*6, signals_to_use= {'keltner': True, 'engulf':True}, capital = 100, trade="long"))
#print(get_keltner_signals(candle_data,candles,1))