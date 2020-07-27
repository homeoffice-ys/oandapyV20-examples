import numpy as np
from moderatebot import PriceTable
from moderatebot import PRecordFactory
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import datetime
import re
from ta import add_all_ta_features
from pyts.decomposition import SingularSpectrumAnalysis


instrument = 'EUR_USD'
granularity = 'M1'
pt = PriceTable(instrument, granularity)
cf = PRecordFactory(pt.granularity)
ask = []
bid = []
t = []
data = []

# to retrieve:
my_file = '/media/office/0D82-9628/data/tick_data/tick_file.json'
with open(my_file, 'r', os.O_NONBLOCK) as f:
    my_list = [json.loads(line) for line in f]

for tick in my_list:
    # print(tick)
    rec = cf.parseTick(tick)
    if rec:
        # t.append(rec[0])
        # tmp = (datetime.datetime(*map(int, re.split('[^\d]', rec[0])[:-1])))
        # print(rec[0])
        tmp = datetime.datetime.strptime(rec[0], '%Y-%m-%dT%H:%M:%S.%fZ')
        # ask.append(rec[1])
        # bid.append(rec[6])
        data.append([tmp, rec[1], rec[6]])
        # data.append([tmp])



df = pd.DataFrame(data, columns=['Time', 'Ask', 'Bid'], dtype=float)
# df = pd.DataFrame(data, columns=['Time'], dtype=str)
print(df.head(5))

# ticks = df.ix[:, ['Ask', 'Bid']]
# bars = ticks.resample('30min').ohlc()
# df.index = pd.to_datetime(df.index, unit='s')
# df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index(pd.to_datetime(df['Time']))
# print(df.head(5))
# exit()

data_ask = df['Ask'].resample('15Min').ohlc()
data_bid = df['Bid'].resample('15Min').ohlc()

print(data_bid.head(5))

'''
For smaller time frames 
(milliseconds/microseconds/seconds), use L for milliseconds, 
U for microseconds, and S for seconds.
'''
df = pd.concat([data_ask, data_bid], axis=1, keys=['Ask', 'Bid'])
print(df.head(5))


