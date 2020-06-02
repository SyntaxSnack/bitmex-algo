#!/usr/bin/env python

import asyncio
import websockets
import pandas as pd
import os.path
from dotenv import load_dotenv
from bitmex_websocket import BitMEXWebsocket

#load constants
load_dotenv()

ws = BitMEXWebsocket(endpoint="https://testnet.bitmex.com/api/v1", symbol="XBTUSD", api_key=os.getenv("bitmex_api_key"), api_secret=os.getenv("bitmex_api_secret"))
