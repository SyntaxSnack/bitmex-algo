import pandas as pd
import numpy as np

from enum import Enum
class Signal(Enum):
    WAIT = 0
    BUY = 1
    SELL = 2

XBTUSD = .5
ETHUSD = .05

#VARIABLES AND DATA
#candles = pd.read_csv("XBTUSD-1m-data.csv", sep=',')
#candle = pd.Series(dtype=object)
    
def candle_df(candles, candleamount):
    candle_data = pd.DataFrame(columns=['open', 'close', 'candle size', 'type'])
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
        candle_data.loc[index] = [data['open'], data['close'], abs(data['open']-data['close']), type]
    print("CANDLE DATA", candle_data)
    return candle_data

#realtime func
def engulfingsignals(curr_row, prev_row, candleamount = 1440, threshold = 1, ignoredoji = False):
    if curr_row[3] == prev_row[3]: #candle type stays the same
        return Signal.WAIT
    elif (curr_row[2] * threshold) > (prev_row[2]) and (ignoredoji == False or prev_row[2] > XBTUSD): # candle is opposite direction and larger
        if curr_row[3] == "red":
            return Signal.SELL
        elif curr_row[3] == "green":
            return Signal.BUY
        else: return Signal.WAIT
    else:
        return Signal.WAIT

#back-test on the series extrapolated from price data
def testengulf(candles, candleamount, threshold=1, ignoredoji=False):
    #first generate a candle-series!
    candle_data = candle_df(candles, candleamount)

    signals=[]
    #generate a trade signal for every candle except for the last, and store in the list we created
    prev_row = candle_data.iloc[1]
    for i,row in candle_data.iloc[1:].iterrows():
        signals.append(engulfingsignals(row, prev_row, candleamount, threshold, ignoredoji))
        prev_row = row
    return(signals)

#run the back-test function
#store into data-frame to compare w/ different engulf parameters
#D=0: including DOJIs in back-test data
#D=1: excluding DOJIs in back-test data
#D=2: both inc and exc in back-test data
min_in_day = 1440
def engulfdata(candleamount=min_in_day, thresholds=[.1,.25,.5, 1], D=2):
    btdata = pd.DataFrame()
    candles = pd.read_csv("XBTUSD-1m-data.csv", sep=',')
    #candles =  candles.iloc[:1000]
    
    if D==0 or D==2:
        print(testengulf(candles, candleamount, thresholds[0]))    
        btdata[candleamount, thresholds[0], "ND"] = testengulf(candles, candleamount, thresholds[0])
        btdata[candleamount, thresholds[1], "ND"] = testengulf(candles, candleamount, thresholds[1])
        btdata[candleamount, thresholds[2], "ND"] = testengulf(candles, candleamount, thresholds[2])
        btdata[candleamount, thresholds[3], "ND"] = testengulf(candles, candleamount, thresholds[3])
    
    if D==1 or D==2:
        btdata[candleamount, thresholds[0], "D"] = testengulf(candles, candleamount, thresholds[0], True)
        btdata[candleamount, thresholds[1], "D"] = testengulf(candles, candleamount, thresholds[1], True)
        btdata[candleamount, thresholds[2], "D"] = testengulf(candles, candleamount, thresholds[2], True)
        btdata[candleamount, thresholds[3], "D"] = testengulf(candles, candleamount, thresholds[3], True)
        
    return(btdata)

#print the responses to the first 50 candlesticks, excluding 0 (WAIT) indicators (for the sake of visibility)
emask = pd.DataFrame(engulfdata() == 0)
edata = pd.DataFrame(engulfdata())[~emask].dropna(how='all')  
btdata = engulfdata(candleamount=1440, thresholds=[.1,.25,.5, 1], D=2)
print('BTDATA', btdata)
#print("# of signals: ",len(edata))
#edata[~emask].dropna(how='all')


#Here we're going to create a dataframe containing all the indicators from the TA library!

#TECHNICAL ANALYSIS INDICATORS | https://github.com/bukosabino/ta | 
import ta
from ta.utils import dropna
# Clean NaN values
df = ta.utils.dropna(candles)

# Initialize Keltner Bands Indicator
indicator_kelt = ta.volatility.KeltnerChannel(high=df["high"], low=df["low"], close=df["close"], n=10, fillna=True)

kseries = pd.Series(dtype=object)

kseries = pd.DataFrame()
# Add Bollinger Bands features
kseries["hband"] = indicator_kelt.keltner_channel_hband_indicator()
kseries["lband"] = indicator_kelt.keltner_channel_lband_indicator()

def keltnersignals(i, engulfsignal):
    if (engulfsignal == 1) and (kseries.iloc[i]["lband"] == 1.0):
        return("BUY")
    elif (engulfsignal == 2) and (kseries.iloc[i]["hband"] == 1.0):
        return("SELL")
    else:
        return("WAIT")

#back-test on the series extrapolated from price data
def keltnertest(candleamount = 1440, threshold = 1, ignoredoji = False):
    ksignals=[]
    for i in candle.index[::-1]:
        engulfsignal = engulfingsignals(i, candleamount, threshold, ignoredoji)
        ksignals.append(keltnersignals(i, engulfsignal))
    return(ksignals)

def keltnerdata(candleamount=1440, thresholds=[.1,.25,.5, 1], D=2):
    btdata = pd.DataFrame()
    if D==0 or D==2:    
        btdata[candleamount, thresholds[0], "ND"] = pd.Series(keltnertest(candleamount, thresholds[0]))
        btdata[candleamount, thresholds[1], "ND"] = pd.Series(keltnertest(candleamount, thresholds[1]))
        btdata[candleamount, thresholds[2], "ND"] = pd.Series(keltnertest(candleamount, thresholds[2]))
        btdata[candleamount, thresholds[3], "ND"] = pd.Series(keltnertest(candleamount, thresholds[3]))
    if D==1 or D==2:
        btdata[candleamount, thresholds[0], "D"] = pd.Series(keltnertest(candleamount, thresholds[0], True))
        btdata[candleamount, thresholds[1], "D"] = pd.Series(keltnertest(candleamount, thresholds[1], True))
        btdata[candleamount, thresholds[2], "D"] = pd.Series(keltnertest(candleamount, thresholds[2], True))
        btdata[candleamount, thresholds[3], "D"] = pd.Series(keltnertest(candleamount, thresholds[3], True))
    return(btdata)

kmask = pd.DataFrame(keltnerdata()) == "WAIT"
kdata = pd.DataFrame(keltnerdata())[~kmask].dropna(how='all')
print("# of signals: ",len(kdata))
kdata