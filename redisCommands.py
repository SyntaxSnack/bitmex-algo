import redis
from walrus import Database  # A subclass of the redis-py Redis client.
import redisCommands as rd
import pandas as pd
import threading
import concurrent.futures
from datetime import datetime

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