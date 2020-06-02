import pandas as pd
import numpy as np

from enum import Enum
class Signal(Enum):
    WAIT = 0
    BUY = 1
    SELL = 2

#VARIABLES AND DATA
candles = pd.read_csv("XBTUSD-1m-data.csv", sep=',')
candle = pd.Series(dtype=object)
    
def candleseries(candleamount):
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
        size = [data['open'], data['close'], abs(data['open']-data['close']), type]
        candle.at[index] = size

#realtime func
def engulfingsignals(candle_idx = -1, candleamount = 1440, threshold = 1, ignoredoji = False):
    print("CANDLE",candle)
    if candle.iloc[candle_idx][2] == candle.iloc[candle_idx-1][2]:
        return Signal.WAIT
    elif (candle.iloc[candle_idx][2] * threshold) > (candle.iloc[candle_idx-1][2]) and (ignoredoji == False or candle.iloc[candle_idx-1][2] > 1):
        if candle.iloc[candle_idx][3] == "red":
            return Signal.SELL
        elif candle.iloc[candle_idx][3] == "green":
            return Signal.BUY
        else: return Signal.WAIT
    else:
        return Signal.WAIT

#back-test on the series extrapolated from price data
def testengulf(candleamount, threshold=1, ignoredoji=False):
    #first generate a candle-series!
    candleseries(candleamount)
    #generate index series for all candles except for the oldest one (no prev candle for it!)
    cseries = iter(candle.index[::-1])
    next(cseries)
    #list to store signals in for easier display
    signals=[]
    #generate a trade signal for every candle except for the last, and store in the list we created
    for i in cseries:
        i = -1*(i)
        signals.append(engulfingsignals(i, candleamount, threshold, ignoredoji))
    return(signals)

#run the back-test function
#store into data-frame to compare w/ different engulf parameters
#D=0: including DOJIs in back-test data
#D=1: excluding DOJIs in back-test data
#D=2: both inc and exc in back-test data
def engulfdata(candleamount=1440, thresholds=[.1,.25,.5, 1], D=2):
    btdata = pd.DataFrame()
    if D==0 or D==2:    
        btdata[candleamount, thresholds[0], "ND"] = testengulf(candleamount, thresholds[0])
        btdata[candleamount, thresholds[1], "ND"] = testengulf(candleamount, thresholds[1])
        btdata[candleamount, thresholds[2], "ND"] = testengulf(candleamount, thresholds[2])
        btdata[candleamount, thresholds[3], "ND"] = testengulf(candleamount, thresholds[3])
    if D==1 or D==2:
        btdata[candleamount, thresholds[0], "D"] = testengulf(candleamount, thresholds[0], True)
        btdata[candleamount, thresholds[1], "D"] = testengulf(candleamount, thresholds[1], True)
        btdata[candleamount, thresholds[2], "D"] = testengulf(candleamount, thresholds[2], True)
        btdata[candleamount, thresholds[3], "D"] = testengulf(candleamount, thresholds[3], True)
    return(btdata)

#print the responses to the first 50 candlesticks, excluding 0 (WAIT) indicators (for the sake of visibility)
emask = pd.DataFrame(engulfdata() == 0)
edata = pd.DataFrame(engulfdata())[~emask].dropna(how='all')
engulfdata(candleamount=1440, thresholds=[.1,.25,.5, 1], D=2)
print("# of signals: ",len(edata))
edata[~emask].dropna(how='all')


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