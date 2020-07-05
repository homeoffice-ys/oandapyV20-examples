import numpy as np
from datetime import datetime
import calendar
import re
import time
from src.moderatebot import PriceTable
from src.moderatebot import PRecordFactory

instrument = 'EUR_USD'
granularity = 'M1'
pt = PriceTable(instrument, granularity)
cf = PRecordFactory(pt.granularity)

index = 1
while index < 2825:

    # tick = np.load('../tick_data/tick_' + str(index) + '.npy', allow_pickle=True)[()]
    tick = np.load('/media/office/0D82-9628/data/tick_data/tick_' + str(index) + '.npy', allow_pickle=True)[()]
    # print(tick)
    rec = []

    if 'PRICE' in tick['type']:
        # print(tick)
        # exit()
        rec = cf.parseTick(tick)


    if rec:
        print('rec ', rec)
        # print(type(rec))
        # print(rec[1])
        # exit()
        pt.addItem(*rec)

    index += 1


