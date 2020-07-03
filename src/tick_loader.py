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

    tick = np.load('../tick_data/tick_' + str(index) + '.npy', allow_pickle=True)[()]

    if 'PRICE' in tick['type']:
        rec = cf.parseTick(tick)

    if rec:
        print('rec ', rec)
        pt.addItem(*rec)

    index += 1


