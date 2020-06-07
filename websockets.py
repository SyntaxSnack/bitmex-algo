#!/usr/bin/env python

import asyncio
import websockets
import pandas as pd
import os.path
from dotenv import load_dotenv
import util
from bitmex_ws.bitmex_websocket import BitMEXWebsocket
import logging
from time import sleep
from pathlib import Path
import pandas as pd
import multiprocessing
from datetime import datetime
import tracemalloc
import redisCommands as rd

#load constants
load_dotenv()
redis_db = []

tracemalloc.start()

# run the websocket
def run(file=True, tradesymbol="XBTUSD", datatype="trades", bid=True, ask=True):  
    #define Walrus data streams on redis
    xstream = rd.xstream()

    #create the BitMex web socket object
    ws = BitMEXWebsocket(endpoint="https://testnet.bitmex.com/api/v1", symbol=tradesymbol, api_key=os.getenv("bitmex_api_key"), api_secret=os.getenv("bitmex_api_secret"))

    logger = setup_logger(file, tradesymbol, datatype)
    logger.info("Instrument data: %s" % ws.get_instrument())

    #run forever
    while(ws.ws.sock.connected):    
            #if ws.api_key:
            #    logger.info("Funds: %s" % ws.funds())
            #logger.info("Market Depth: %s" % ws.market_depth())
            #logger.info("Recent Trades: %s\n\n" % ws.recent_trades())
            #logger.info("Executable price: %s\n\n" % ws.executableprice())
            
            # `ignore_index=True` has to be provided, otherwise you'll get
            # "Can only append a Series if ignore_index=True or if the Series has a name" errors
            df = pd.DataFrame()
            df = df.append(ws.executableprice(), ignore_index=True)
            x = rd.ExecutablePrice()
            x.bids = list(df.bids.map(lambda x: x[0]))
            x.asks = list(df.asks.map(lambda x: x[0]))
            if bid:
                print("Bid sent")
                xstream.xbid.add({'price': x.bids[0][0], 'amount': x.bids[0][1]})
                #xstream.xbid.add({'timestamp': x.timestamp})
            if ask:
                print("Ask sent")
                xstream.xask.add({'price': x.asks[0][0], 'amount': x.asks[0][1]})
                #xstream.xask.add({'timestamp': x.timestamp})

def setup_logger(file, tradesymbol, datatype):
    logger = logging.getLogger()
    if not file:
        # Prints logger info to terminal
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)  # Change this to DEBUG if you want a lot more info
        ch = logging.StreamHandler()
        # create formatter
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        # add formatter to ch
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        # Prints logger info to CSV file
        logger.setLevel(logging.INFO)  # Change this to DEBUG if you want a lot more info
        projectpath = Path(__file__).parent.absolute()
        data_folder = Path(projectpath,"OrderBookData",tradesymbol, datatype + "." + "csv")
        ch = logging.FileHandler(data_folder)
        # create formatter
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        # add formatter to ch
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

#if __name__ == "__main__":
#    p = multiprocessing.Pool()
#    result = p.imap_unordered(run, [True, False])
#    #terminate process on key press
#    stop_char=""
#    while stop_char.lower() != "q":
#        stop_char=input("Enter 'q' to quit ")
#    print("terminate process")
#    p.terminate()

run(True)