import redis
from walrus import Database  # A subclass of the redis-py Redis client.
import redisCommands as rd
import pandas as pd
import threading
import concurrent.futures

class ExecutablePrice:
    bids: list
    asks: list
    #We don't need this because walrus package time-series stream automatically appends a timestamp!
    #timestamp: str
        
    #constructor 
    #def __init__(self): 
    #    self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

class xMsg:
    price: list
    amount: list
    timestamp: datetime
    #We don't need this because walrus package time-series stream automatically appends a timestamp!
    #timestamp: str
        
    #constructor 
    #def __init__(self): 
    #    self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

def startRedis(host='localhost', port=6379, db=0):
    return(redis.StrictRedis(host=host, port=port, db=db))

def xstream(timeseries=True, name=None):
    db = Database()
    if timeseries:
        return(db.time_series('ExecutablePrice', ['xbid', 'xask']))
    else:
        return(db.Stream(name))

def wait_until(func):
    while(True):
        s = stream.xbid.read()
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
    data = xMsg()
    data.price = message.data['price']
    data.amount = message.data['amount']
    data.timestamp = message.timestamp
    #df = pd.DataFrame(data, columns=["price", "amount", "timestamp"])
    return(data)

#example of polling through the latest price data with it being a data object
#while(True):
#    data = wait_until_par(read_xstream)
#    print(data.price, data.amount, data.timestamp)