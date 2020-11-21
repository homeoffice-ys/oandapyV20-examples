import numpy as np
from moderatebot import PriceTable
from moderatebot import PRecordFactory
import matplotlib.pyplot as plt
import json
import os
from pyts.decomposition import SingularSpectrumAnalysis


instrument = 'EUR_USD'
granularity = 'M1'
pt = PriceTable(instrument, granularity)
cf = PRecordFactory(pt.granularity)
ask = []
bid = []
t = []

# to retrieve:
my_file = '/media/office/0D82-9628/data/tick_data/tick_file_2.json'
with open(my_file, 'r', os.O_NONBLOCK) as f:
    my_list = [json.loads(line) for line in f]

for tick in my_list:
    # print(tick)
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
