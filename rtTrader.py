import redis
import redisCommands as rd
import pandas as pd
import threading
import concurrent.futures
from datetime import datetime
import redisCommands as rd
import time
import random

#Realtime trading parameters
symbol = 'XBTUSD'
qty = -1

def wait_until(func):
    while(True):
        s = xstream.xbid.read()
        for message in s:
            if hasattr(message, 'stream'):
                return(func(message))

def wait_until_par(*args):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(wait_until, *args)
        return_value = future.result()
        return(return_value)

    #t = threading.Thread(target=wait_until, args=args)
    #t.start()
    #return(t)
      
#IF YOU WANT TO GET DATA W/O DELAY BY USING THIS FUNCTION, USE THE FOLLOWING:
#data = wait_until_par(read_xstream)
def read_xstream(message):
    data = rd.xMsg()
    data.price = message.data['price']
    data.amount = message.data['amount']
    data.timestamp = message.timestamp
    #df = pd.DataFrame(data, columns=["price", "amount", "timestamp"])
    return(data)

xstream = rd.xstream()

#example of polling through the latest price data with it being a data object
#while(True):
#    data = wait_until_par(read_xstream)
#    print(data.price, data.amount, data.timestamp)

#variables to replace later with trade logic
'''
buy = []
sell = []
for i in range(100):
    buy.append(bool(random.getrandbits(1)))
    sell.append(bool(random.getrandbits(1)))
'''

def getSignal(price, amount):
    buy = bool(random.getrandbits(1))
    sell = bool(random.getrandbits(1))

    return('buy')

while(True):
    data = wait_until_par(read_xstream)

    price = int(data.price)
    amount = int(data.amount)


    print(data.price, data.amount, data.timestamp)

    if (getSignal(int(price), int(amount)) == 'buy')
        order = bitmex_client.Order.Order_new(symbol, orderQty=qty, price=price).result()
print(order_result)
orders = bitmex_client.Order.Order_getOrders().result()[0]
        print("TRADE")



    #Print all our current orders
    for order in orders:
        print(order)
        processed_order = {}
        processed_order["symbol"] = order["symbol"]
        processed_order["amount"] = str(order["orderQty"]).split("L")[0]
        processed_order["price"] = order["price"]
        processed_order["side"] = order["side"]
        processed_order["status"] = order["ordStatus"]
        print(processed_order)
    time.sleep(1)