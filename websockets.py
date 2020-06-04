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
import asyncio

#app = faust.App('myapp', broker='kafka://localhost')

# Models describe how messages are serialized:
# {"account_id": "3fae-...", amount": 3}
#class Order(faust.Record):
#    account_id: str
#   amount: int

#@app.agent(value_type=Order)
#async def order(orders):
#    async for order in orders:
        # process infinite stream of orders.
#        print(f'Order for {order.account_id}: {order.amount}')

#load constants
load_dotenv()

# run the websocket
async def run(file=True, tradesymbol="XBTUSD", datatype="trades"):  
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
            logger.info("Executable price: %s\n\n" % ws.executableprice())


def setup_logger(file, tradesymbol, datatype):
    logger = logging.getLogger()
    if not file:
        # Prints logger info to terminal
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)  # Change this to DEBUG if you want a lot more info
        ch = logging.StreamHandler()
        # create formatter
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
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
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        # add formatter to ch
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


if __name__ == "__main__":
    asyncio.run(run(file=False))
    asyncio.run(run(file=True))