import numpy as np
from datetime import datetime
import calendar
import re
import time
from moderatebot import PriceTable
from moderatebot import PRecordFactory
import matplotlib.pyplot as plt


instrument = 'EUR_USD'
granularity = 'M1'
pt = PriceTable(instrument, granularity)
cf = PRecordFactory(pt.granularity)
ask = []
bid = []
t = []

index = 1
while index < 2825:

    # tick = np.load('../tick_data/tick_' + str(index) + '.npy', allow_pickle=True)[()]
    tick = np.load('/media/office/0D82-9628/data/tick_data/July3_1.5hrs/tick_' + str(index) + '.npy', allow_pickle=True)[()]
    print(tick)
    exit()
    rec = []

    if 'PRICE' in tick['type']:
        # print(tick)
        # exit()
        rec = cf.parseTick(tick)
        # print('rec ', rec)
        # print(type(rec))
        t.append(rec[0])
        ask.append(rec[1])
        bid.append(rec[6])
        # print(len(rec))
        # exit()
        pt.addItem(*rec)

    index += 1

# fig, ax = plt.subplots()
plt.plot(t, np.array(ask), label='ask')
plt.plot(t, bid, label='bid')
# plt.plot(t, (np.array(ask) - np.array(bid))+1.124, label='spread')
ax = plt.gca()
ax.get_figure().autofmt_xdate()
ax.set_xticks(ax.get_xticks()[::5])
plt.grid(True)
plt.legend()
plt.show()
