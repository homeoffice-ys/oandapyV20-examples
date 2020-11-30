import numpy as np
from moderatebot import PriceTable
from moderatebot import PRecordFactory
import matplotlib.pyplot as plt
import json
import os
from pyts.decomposition import SingularSpectrumAnalysis
import pandas as pd
import my_utils as mu
from datetime import datetime, time, timedelta
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000


instrument = 'EUR_USD'
granularity = 'M1'
pt = PriceTable(instrument, granularity)
cf = PRecordFactory(pt.granularity)
ask = []
bid = []
t = []

# to retrieve:
my_file = '/home/john/Documents/data/tick_data/tick_file_6.json'
my_file = '../../../data/tick_data/tick_file_2.json'
pth = '../../../data/tick_data'

# tick = []
bid = []
ask = []
avg = []
aohlc = {}
bohlc = {}
candles = []
for file in range(7):
    print('loading ', os.path.join(pth, 'tick_file_' + str(file + 1) + '.json'))
    for line in open(os.path.join(pth, 'tick_file_' + str(file+1) + '.json'), 'r'):
        tick = json.loads(line)
        # print(tick)
        # exit()
        # since tick data time stamp has nano second format: 2020-07-05T23:36:02.009657758Z
        # the nano seconds are dropped and the datetime is appended with only microsecond accuracy
        # i.e., 2020-07-05T23:36:02.009657758Z -> 2020-07-05T23:36:02.009657
        if len(t) == 0:
            aohlc['open'] = tick['asks'][0]['price']
            bohlc['open'] = tick['bids'][0]['price']
            aohlc['high'] = tick['asks'][0]['price']
            bohlc['high'] = tick['bids'][0]['price']
            aohlc['low'] = tick['asks'][0]['price']
            bohlc['low'] = tick['bids'][0]['price']
            t.append(datetime.strptime(tick['time'][0:26], "%Y-%m-%dT%H:%M:%S.%f"))

        elif datetime.strptime(tick['time'][0:26], "%Y-%m-%dT%H:%M:%S.%f") - t[0] <\
                datetime.strptime('0:5:0', '%H:%M:%S') - datetime.strptime('0:0:0', '%H:%M:%S'):
            aohlc['high'] = max(aohlc['high'], tick['asks'][0]['price'])
            bohlc['high'] = max(bohlc['high'], tick['bids'][0]['price'])
            aohlc['low'] = min(aohlc['low'], tick['asks'][0]['price'])
            bohlc['low'] = min(bohlc['low'], tick['bids'][0]['price'])

        else:
            aohlc['high'] = max(aohlc['high'], tick['asks'][0]['price'])
            bohlc['high'] = max(bohlc['high'], tick['bids'][0]['price'])
            aohlc['low'] = min(aohlc['low'], tick['asks'][0]['price'])
            bohlc['low'] = min(bohlc['low'], tick['bids'][0]['price'])
            aohlc['close'] = tick['asks'][0]['price']
            bohlc['close'] = tick['bids'][0]['price']
            # print({'time': tick['time'][0:26]})
            # exit()
            # candles.append([{'time': tick['time'][0:26]}, bohlc, aohlc])
            candles.append([tick['time'][0:26],
                            bohlc['open'], bohlc['high'], bohlc['low'], bohlc['close'],
                            aohlc['open'], aohlc['high'], aohlc['low'], aohlc['close']])
            # print(candles)
            # exit()
            aohlc = {}
            bohlc = {}
            t = []
        # print(tmp['bids'])
        # print(tmp['closeoutBid'])
        # print(tmp['closeoutAsk'])
        bid.append(float(tick['closeoutBid']))
        ask.append(float(tick['closeoutAsk']))
        avg.append((bid[-1]+ask[-1])/2)
        # dat = (float(tmp['closeoutBid']) + float(tmp['closeoutAsk'])) / 2
        # dat = (dat - 1.12) * 10000
        # print(type(dat))

        # tick.append(dat)
    # print(candles[0])
    # print(candles[1])
    # print(candles[2])
    # exit()
print(len(candles))
print(candles[0])
print(candles[-1])
np = len(bid)
print(np)
# pips = tick - min(tick)
#
# pips = []
min_bid = min(bid)
min_ask = min(ask)
min_avg = min(avg)

pips_bid = [(x - min_bid)*10**4 for x in bid]
pips_ask = [(x - min_ask)*10**4 for x in ask]
pips_avg = [(x - min_avg)*10**4 for x in avg]
spread = [(x - y)*10**4 for x, y in zip(ask, bid)]

# spread = [0 if (x < 0) else x for x in spread]

del bid, ask, avg
# for x in tick:
#     pips.append((x-mn)*10000)
# plt.plot(pips_bid)

plt.plot(pips_ask)
plt.figure()
plt.plot(spread, '.')
# plt.errorbar(range(np), pips_avg, spread)
plt.show()



# save the plot as a file
# fig.savefig('two_different_y_axis_for_single_python_plot_with_twinx.jpg',
#             format='jpeg',
#             dpi=100,
#             bbox_inches='tight')

exit()
    # print(tick)
    # print(len(tick))
    # train = pd.DataFrame.from_dict(tick, orient='index')
    # train.reset_index(level=0, inplace=True)
    # print(train.shape[0])
    # exit()
    # dict_train = json.loads(line)

print(train.shape[0])
exit()


# with open(my_file) as train_file:
#     dict_train = json.load(train_file)

# converting json dataset from dictionary to dataframe
train = pd.DataFrame.from_dict(dict_train, orient='index')
train.reset_index(level=0, inplace=True)

print(train)
print(type(train))
total_rows = train.shape[0]
print(total_rows + 1)

exit()

for tick in my_list:
    print(tick)
    exit()
    rec = cf.parseTick(tick)
    if rec:
        t.append(rec[0])
        ask.append(rec[1])
        bid.append(rec[6])
        # print(len(rec))
        # exit()
        # pt.addItem(*rec)

# fig, ax = plt.subplots()
t = None
bid = None
my_list = None
print(len(ask))
plt.plot(ask)
# plt.plot(t, ask, label='ask')
# plt.plot(t, bid, label='bid')
# plt.plot(t, (np.array(ask) - np.array(bid)) + 1.124, label='spread')
# ax = plt.gca()
# ax.get_figure().autofmt_xdate()
# ax.set_xticks(ax.get_xticks()[::5000])
plt.grid(True)
# plt.legend()
plt.show()
#
# # Parameters
# n_samples, n_timestamps = 100, 48
#
# # Toy dataset
# rng = np.random.RandomState(41)
# X = rng.randn(n_samples, n_timestamps)
#
# # We decompose the time series into three subseries
# window_size = 15
# groups = [np.arange(i, i + 5) for i in range(0, 11, 5)]
#
# # Singular Spectrum Analysis
# ssa = SingularSpectrumAnalysis(window_size=15, groups=groups)
# X_ssa = ssa.fit_transform(X)
#
# # Show the results for the first time series and its subseries
# plt.figure(figsize=(16, 6))
#
# ax1 = plt.subplot(121)
# ax1.plot(X[0], 'o-', label='Original')
# ax1.legend(loc='best', fontsize=14)
#
# ax2 = plt.subplot(122)
# for i in range(len(groups)):
#     ax2.plot(X_ssa[0, i], 'o--', label='SSA {0}'.format(i + 1))
# ax2.legend(loc='best', fontsize=14)
#
# plt.suptitle('Singular Spectrum Analysis', fontsize=20)
#
# plt.tight_layout()
# plt.subplots_adjust(top=0.88)
# plt.show()
