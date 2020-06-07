#!/usr/bin/env python

# IMPORTS
import pandas as pd
from bitmex import bitmex
import requests, json
import pytz # $ pip install pytz
from dotenv import load_dotenv
import math
import os.path
import time
from datetime import timedelta, datetime
from dateutil import parser
from tqdm import tqdm_notebook #(Optional, used for progress-bars)

#load constants
load_dotenv()

#create the bitmex client object
bitmex_client = bitmex(test=os.getenv("TESTNET"), api_key=os.getenv("bitmex_api_key"), api_secret=os.getenv("bitmex_api_secret"))

#converts start_time and end_time to datetime objects
def get_datetimes(symbol, kline_size, start_time, end_time, data, source):

    #if len(data) > 0:  old = parser.parse(data["timestamp"].iloc[-1]).replace(tzinfo=None)
    if source == "binance":
        if start_time == None:
            old = datetime.strptime('1 Jan 2017', '%d %b %Y')
        else:
            old = datetime.strptime(start_time, '%d %b %Y')
    elif source == "bitmex":
        if start_time == None:
            old = bitmex_client.Trade.Trade_getBucketed(symbol=symbol, binSize=kline_size, count=1, reverse=False).result()[0][0]['timestamp']
            old = old.replace(tzinfo=None)
        else:
            old = datetime.strptime(start_time, '%d %b %Y')
            print("GOT HEERERE")
    print("old time", old)

    if source == "binance": 
        if end_time == 'PRESENT':
            new = pd.to_datetime(binance_client.get_klines(symbol=symbol, interval=kline_size)[-1][0], unit='ms')
        else:
            new = datetime.strptime(end_time, '%d %b %Y')

    elif source == "bitmex":
        if end_time == 'PRESENT':
            new = bitmex_client.Trade.Trade_getBucketed(symbol=symbol, binSize=kline_size, count=1, reverse=True).result()[0][0]['timestamp']
            new = new.replace(tzinfo=None)
        else:
            new = datetime.strptime(end_time, '%d %b %Y')
    return old, new

def get_all_bitmex(symbol, kline_size, start_time, end_time, save = False):
    filename = '%s-%s-data.csv' % (symbol, kline_size)
    print(filename)
    if os.path.isfile(filename): data_df = pd.read_csv(filename)
    else: data_df = pd.DataFrame()

    oldest_point, newest_point = get_datetimes(symbol,kline_size, start_time, end_time,  data_df, source = "bitmex")
    print("getting data from", oldest_point, " to ", newest_point)
    count_minutes = int(divmod((newest_point - oldest_point).total_seconds(),60)[0])

    ranges = []
    for i in range(0,count_minutes,1000):
        if(i+1000 > count_minutes):
            ranges.append([i,count_minutes])
        else: ranges.append([i, i+999])
    #MAXIMUM count is 1000 so we need to loop until we hit the value, adding 1000
    # We can only get 1000 values at a time, so we go from newest to oldest point by intervals of 1000 or less
    for r in ranges:
            start = oldest_point + timedelta(minutes=r[0])
            end = oldest_point + timedelta(minutes=r[1])
            data = bitmex_client.Trade.Trade_getBucketed(symbol=symbol, binSize=kline_size, count=r[1] - r[0], startTime = start, endTime = end).result()[0]
            temp_df = pd.DataFrame(data)
            data_df = data_df.append(temp_df)

    data_df.set_index('timestamp', inplace=True)
    if save : data_df.to_csv(filename)
    print('All caught up..!')
    return data_df

#CONVERTING TIME TO TZ AWARE OBJECT
# convert the time string to a datetime object
dt_str = "5/30/2020 4:05:03:10:10"
#unaware_est = datetime.strptime(dt_str,"%m/%d/%Y %H:%M:%S+00:00")
# make it a timezone-aware datetime object 
#est_time = pytz.timezone('US/Eastern').localize(unaware_est, is_dst=None)

#Get timeseries data
#data = get_all_bitmex(symbol="XBTUSD", kline_size="1m", start_time='29 May 2020', end_time='PRESENT', save=True)
data = get_all_bitmex(symbol="XBTUSD", kline_size="1m", start_time='1 Jan 2019', end_time='PRESENT', save=True)
print(data)

#Price data is just public.
ether = requests.get("https://testnet.bitmex.com/api/v1/orderBook/L2?symbol=ETHUSD&depth=1").json()
xbt = requests.get("https://testnet.bitmex.com/api/v1/orderBook/L2?symbol=xbt&depth=1").json()
ether_ask_price = ether[0]['price']
ether_bid_price = ether[1]['price']
print(ether_ask_price)
print(ether_bid_price)

#I think this only works for certain kinds of requests (like those listed here - https://testnet.bitmex.com/api/explorer/)
#GET /orderBook/L2

xbt = requests.get("https://testnet.bitmex.com/api/v1/orderBook/L2?symbol=xbt&depth=1").json()
#a = client.Quote.Quote_get(symbol="XBTUSD", startTime=datetime.datetime(2018, 1, 1)).result()
#print(a)

symbol = 'XBTUSD'
qty = -1
price = ether[1]['price']

order_result = bitmex_client.Order.Order_new(symbol=symbol, orderQty=qty, price=price).result()
print(order_result)
orders = bitmex_client.Order.Order_getOrders().result()[0]

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