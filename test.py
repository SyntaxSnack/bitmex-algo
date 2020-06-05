from datetime import datetime
from pathlib import Path

time = datetime.now().strftime("%Y%m%d-%H%M%S")
backtestfile = Path("Backtest",time + ".txt")
f = open(backtestfile, "w")
f.write("hey")