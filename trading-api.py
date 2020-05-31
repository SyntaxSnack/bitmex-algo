from bitmex import bitmex
import requests, json
import pytz


# IMPORTS
import pytz # $ pip install pytz
import pandas as pd
import math
import os.path
import time
from datetime import timedelta, datetime
from dateutil import parser
from tqdm import tqdm_notebook #(Optional, used for progress-bars)



### API
bitmex_api_key = 'tZzFtVRt9SHzcsIKeLZpckt3'
bitmex_api_secret = 'G8Ln9_0mC3wvD_lE61Yji_wDByxkjuv2scKRJgRu06-2lv50'

### CONSTANTS
binsizes = {"1m": 1, "5m": 5, "1h": 60, "1d": 1440}
batch_size = 750
bitmex_client = bitmex(test=True, api_key=bitmex_api_key, api_secret=bitmex_api_secret)

### FUNCTIONS
def minutes_of_new_data(symbol, kline_size, start_time, end_time, data, source):

    if len(data) > 0:  old = parser.parse(data["timestamp"].iloc[-1])
    elif source == "binance":
        if start_time == None:
            old = datetime.strptime('1 Jan 2017', '%d %b %Y')
        else:
            old = datetime.strptime(start_time, '%d %b %Y')
    
    elif source == "bitmex":
        if start_time == None:
            old = bitmex_client.Trade.Trade_getBucketed(symbol=symbol, binSize=kline_size, count=1, reverse=False).result()[0][0]['timestamp']
        else:
            old = datetime.strptime(start_time, '%d %b %Y')
    if source == "binance": 
        if end_time == 'PRESENT':
            new = pd.to_datetime(binance_client.get_klines(symbol=symbol, interval=kline_size)[-1][0], unit='ms')
        else:
            new = datetime.strptime(end_time, '%d %b %Y')
    if source == "bitmex":
        if end_time == 'PRESENT':
            new = bitmex_client.Trade.Trade_getBucketed(symbol=symbol, binSize=kline_size, count=1, reverse=True).result()[0][0]['timestamp']
            new = new.replace(tzinfo=None)
        else:
            new = datetime.strptime(end_time, '%d %b %Y')
            print(new)
    return old, new

def get_all_bitmex(symbol, kline_size, start_time, end_time, save = False):
    filename = '%s-%s-data.csv' % (symbol, kline_size)
    if os.path.isfile(filename): data_df = pd.read_csv(filename)
    else: data_df = pd.DataFrame()
    #if(oldest_point == None):
    print("This is start time:",start_time)
    print("This is end time ",end_time)
    oldest_point, newest_point = minutes_of_new_data(symbol,kline_size, start_time, end_time,  data_df, source = "bitmex")
    print("This is oldest point:",oldest_point)

    print("This is newest point",newest_point)
    delta_min = (newest_point - oldest_point).total_seconds()/60
    available_data = math.ceil(delta_min/binsizes[kline_size])
    rounds = math.ceil(available_data / batch_size)
    if rounds > 0:
        print('Downloading %d minutes of new data available for %s, i.e. %d instances of %s data in %d rounds.' % (delta_min, symbol, available_data, kline_size, rounds))
        for round_num in tqdm_notebook(range(rounds)):
            time.sleep(1)
            new_time = (oldest_point + timedelta(minutes = round_num * batch_size * binsizes[kline_size]))
            print(new_time)
            data = bitmex_client.Trade.Trade_getBucketed(symbol=symbol, binSize=kline_size, count=batch_size, startTime = new_time).result()[0]
            temp_df = pd.DataFrame(data)
            data_df = data_df.append(temp_df)
    data_df.set_index('timestamp', inplace=True)
    if save and rounds > 0: data_df.to_csv(filename)
    print('All caught up..!')
    return data_df

#CONVERTING TIME TO TZ AWARE OBJECT
# convert the time string to a datetime object
dt_str = "5/30/2020 4:05:03:10:10"
#unaware_est = datetime.strptime(dt_str,"%m/%d/%Y %H:%M:%S+00:00")
# make it a timezone-aware datetime object 
#est_time = pytz.timezone('US/Eastern').localize(unaware_est, is_dst=None)

#Get timeseries data
data = get_all_bitmex(symbol="XBTUSD", kline_size="1m", start_time='1 Jan 2017', end_time='PRESENT', save=True)
print(data)

#Price data is just public.
ether = requests.get("https://testnet.bitmex.com/api/v1/orderBook/L2?symbol=ETHUSD&depth=1").json()
xbt = requests.get("https://testnet.bitmex.com/api/v1/orderBook/L2?symbol=xbt&depth=1").json()
ether_ask_price = ether[0]['price']
ether_bid_price = ether[1]['price']
print(ether_ask_price)
print(ether_bid_price)


client = bitmex.bitmex(test=True, api_key=bitmex_api_key, api_secret=bitmex_api_secret)
#I think this only works for certain kinds of requests (like those listed here - https://testnet.bitmex.com/api/explorer/)
#GET /orderBook/L2

xbt = requests.get("https://testnet.bitmex.com/api/v1/orderBook/L2?symbol=xbt&depth=1").json()
#a = client.Quote.Quote_get(symbol="XBTUSD", startTime=datetime.datetime(2018, 1, 1)).result()
#print(a)

symbol = 'ETHUSD'
qty = -1
price = ether[1]['price']

order_result = client.Order.Order_new(symbol=symbol, orderQty=qty, price=price).result()
print(order_result)
orders = client.Order.Order_getOrders().result()[0]

for order in orders:
   print(order)
   processed_order = {}
   processed_order["symbol"] = order["symbol"]
   processed_order["amount"] = str(order["orderQty"]).split("L")[0]
   processed_order["price"] = order["price"]
   processed_order["side"] = order["side"]
   processed_order["status"] = order["ordStatus"]
   print(processed_order)
#https://testnet.bitmex.com/api/v1/orderBook/L2?symbol=XBT

#Next step is to try creating orders

