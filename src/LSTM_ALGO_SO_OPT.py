import numpy as np
from moderatebot import PriceTable
from moderatebot import PRecordFactory
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import datetime
from ta import add_all_ta_features
import ta
from pyts.decomposition import SingularSpectrumAnalysis

def Stoch(close,high,low):
    slowk, slowd = ta.STOCH(high, low, close)
    stochSell = ((slowk < slowd) & (slowk.shift(1) > slowd.shift(1))) & (slowd > 80)
    stochBuy = ((slowk > slowd) & (slowk.shift(1) < slowd.shift(1))) & (slowd < 20)
    return stochSell, stochBuy, slowk, slowd

instrument = 'EUR_USD'
granularity = 'M1'
pt = PriceTable(instrument, granularity)
cf = PRecordFactory(pt.granularity)
data = []

# to retrieve:
my_file = '/media/office/0D82-9628/data/tick_data/tick_file.json'
with open(my_file, 'r', os.O_NONBLOCK) as f:
    my_list = [json.loads(line) for line in f]

for tick in my_list:
    # print(tick)
    rec = cf.parseTick(tick)
    if rec:
        tmp = datetime.datetime.strptime(rec[0], '%Y-%m-%dT%H:%M:%S.%fZ')
        price = (rec[1] + rec[6])/2
        spread = (rec[1] - rec[6])
        # data.append([tmp, rec[1], rec[6]])
        data.append([tmp, price, spread])

df = pd.DataFrame(data, columns=['Time', 'Price', 'Spread'], dtype=float)
df = df.set_index(pd.to_datetime(df['Time']))
data_price = df['Price'].resample('15Min').ohlc()
'''
For smaller time frames 
(milliseconds/microseconds/seconds), use L for milliseconds, 
U for microseconds, and S for seconds.
'''
# df = pd.concat([data_price], axis=1, keys=['Price'])
df = pd.concat([data_price], axis=1)
print(df.head(5))
print(df['high'])
'''
class
    ta.momentum.StochasticOscillator(
    high: pandas.core.series.Series, 
    low: pandas.core.series.Series, 
    close: pandas.core.series.Series, 
    n: int = 14, d_n: int = 3, fillna: bool = False)
'''
SO = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], n=14, d_n=3)
print(dir(SO))
print(vars(SO))
print(SO.__dict__)

