# -*- coding: utf-8 -*-
# ! /usr/bin/env python3

import re
import time
import argparse
from datetime import datetime
import calendar
import json
import logging
from oandapyV20 import API
from oandapyV20.exceptions import V20Error
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.positions as positions
from oandapyV20.contrib.requests import (
    MarketOrderRequest,
    TakeProfitDetails,
    StopLossDetails
)
# import pandas as pd
import numpy as np
import sys
# from oandapyV20.definitions.instruments import CandlestickGranularity
import oandapyV20.definitions.instruments as defs
from src.exampleauth import exampleAuth

""" Simple trading application based on MovingAverage crossover.

    Positions can be tracked using the oanda_console demo program.

    **********************************************************
    * THIS PROGRAM IS SOLELY MEANT FOR DEMONSTRATION PURPOSE! *
    * NEVER US THIS ON A LIVE ACCOUNT                        *
    **********************************************************

    - The BotTrader class creates a PriceTable for the instrument.
    - A MovingAverage - crosssover indicator, MAx, is added and
      attached to the pricetable. Each time the pricetable gets a new
      record added and 'onAddItem' event is triggered which has the
      MAx calculate method attached.
    - if MAx has a state change LONG -> SHORT or SHORT -> LONG
      a marketorder is created with a stoploss and a takeprofit
      before placing the new order existing positions are closed
    - check the logfile to trace statechanges, orders, etc.


"""

logging.basicConfig(
    filename="./simplebot.log",
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s : %(message)s',
)

logger = logging.getLogger(__name__)


NEUTRAL = 0
SHORT = 1
LONG = 2


def mapstate(s):
    states = {
       NEUTRAL: "NEUTRAL",
       SHORT: "SHORT",
       LONG: "LONG",
    }
    return states[s]


class Event(object):

    def __init__(self):
        self.handlers = set()

    def handle(self, handler):
        logger.info("%s: adding handler: %s",
                    self.__class__.__name__, handler.__name__)
        self.handlers.add(handler)
        return self

    def unhandle(self, handler):
        try:
            self.handlers.remove(handler)
        except:
            raise ValueError("Handler is not handling this event, "
                             "so cannot unhandle it.")
        return self

    def fire(self, *args, **kargs):
        for handler in self.handlers:
            handler(*args, **kargs)

    def getHandlerCount(self):
        return len(self.handlers)

    __iadd__ = handle
    __isub__ = unhandle
    __call__ = fire
    __len__ = getHandlerCount

class Indicator(object):
    """indicater baseclass."""
    def __init__(self, pt):
        self._pt = pt
        self.values = [None] * len(self._pt._dt)

    def calculate(self):
        raise Exception("override this method")

    def __len__(self):
        return len(self._pt)

    def __getitem__(self, i):
        def rr(_i):
            if _i >= len(self._pt):  # do not go beyond idx
                raise IndexError("list assignment index out of range")
            if _i < 0:
                _i = self._pt.idx + _i

            return self.values[_i]

        if isinstance(i, int):
            return rr(i)
        elif isinstance(i, slice):
            return [rr(j) for j in range(*i.indices(len(self)))]
        else:
            raise TypeError("Invalid argument")

class MAx(Indicator):
    """Moving average crossover."""

    def __init__(self, pt, smaPeriod, lmaPeriod, pK, pD, pF):
        super(MAx, self).__init__(pt)
        self.smaPeriod = smaPeriod
        self.lmaPeriod = lmaPeriod
        self.pK = pK
        self.pD = pD
        self.pF = pF
        self._events = Event()
        self.state = NEUTRAL

    def calculate(self, idx):
        print('idx in calculate ', idx)
        if idx <= self.lmaPeriod:   # not enough values to calculate MAx
            self.values[idx-1] = None
            return
        o = (self._pt._ao + self._pt._bo) / 2
        h = (self._pt._ah + self._pt._bh) / 2
        l = (self._pt._al + self._pt._bl) / 2
        c = (self._pt._ac + self._pt._bc) / 2

        # perform inefficient MA calculations to get the MAx value
        SMA = sum(c[idx-self.smaPeriod:idx]) / self.smaPeriod
        LMA = sum(c[idx-self.lmaPeriod:idx]) / self.lmaPeriod

        # fast SO
        K = (c[idx] - np.min(l[idx-self.pK:idx])) /(np.max(h[idx-self.pK:idx]) - np.min(l[idx-self.pK:idx])) * 100
        D = np.mean(K[idx-self.pD:idx])
        # slow SO
        # KS = D
        # DS = MA(KS,3)
        # full SO
        # KF = MA(K,p2)
        # DF = MA(KF,p3)

        self.values[idx-1] = SMA - LMA
        # self.state = LONG if self.values[idx-1] > 0 else SHORT
        if (self.values[idx - 1] > 0) and (K > 80) and (D < K):
            self.state = LONG
        elif (self.values[idx - 1] < 0) and (K < 20) and (D > K):
            self.state = SHORT

        logger.info("MAx: processed %s : state: %s",
                    self._pt[-1][0], mapstate(self.state))

class PriceTable(object):

    def __init__(self, instrument, granularity):
        self.instrument = instrument
        self.granularity = granularity
        self._dt = [None] * 1000  # allocate space for datetime
        self._ac = [None] * 1000   # allocate space for close values
        self._av = [None] * 1000   # allocate space for volume values
        self._ao = [None] * 1000  # allocate space for open values
        self._ah = [None] * 1000  # allocate space for high values
        self._al = [None] * 1000  # allocate space for low values
        self._bc = [None] * 1000   # allocate space for close values
        self._bv = [None] * 1000   # allocate space for volume values
        self._bo = [None] * 1000  # allocate space for open values
        self._bh = [None] * 1000  # allocate space for high values
        self._bl = [None] * 1000  # allocate space for low values


        self._events = {}         # registered events
        self.idx = 0

    def fireEvent(self, name, *args, **kwargs):
        if name in self._events:
            f = self._events[name]
            f(*args, **kwargs)

    def setHandler(self, name, f):
        if name not in self._events:
            self._events[name] = Event()
        self._events[name] += f

    def addItem(self, dt, ao, ah, al, ac, av, bo, bh, bl, bc, bv):
        self._dt[self.idx] = dt
        self._ac[self.idx] = ac
        self._av[self.idx] = av
        self._ao[self.idx] = ao
        self._ah[self.idx] = ah
        self._al[self.idx] = al
        self._bc[self.idx] = bc
        self._bv[self.idx] = bv
        self._bo[self.idx] = bo
        self._bh[self.idx] = bh
        self._bl[self.idx] = bl

        self.idx += 1
        self.fireEvent('onAddItem', self.idx)

    def __len__(self):
        return self.idx

    def __getitem__(self, i):
        def rr(_i):
            if _i >= self.idx:  # do not go beyond idx in the reserved items
                raise IndexError("list assignment index out of range")
            if _i < 0:
                _i = self.idx + _i   # the actual end of the array
            return (self._dt[_i], self._ac[_i], self._av[_i], self._ao[_i], self._ah[_i], self._al[_i],
                    self._bc[_i], self._bv[_i], self._bo[_i], self._bh[_i], self._bl[_i])

        if isinstance(i, int):
            return rr(i)
        elif isinstance(i, slice):
            return [rr(j) for j in range(*i.indices(len(self)))]
        else:
            raise TypeError("Invalid argument")

class PRecordFactory(object):
    """generate price records from streaming prices."""
    def __init__(self, granularity):
        self._last = None
        self._granularity = granularity
        self.interval = self.granularity_to_time(granularity)
        self.ask_data = {"o": None, "h": None, "l": None, "c": None, "v": 0}
        self.bid_data = {"o": None, "h": None, "l": None, "c": None, "v": 0}
        self.ask = []
        self.bid = []

    def parseTick(self, t):

        rec = None
        if not self._last:
            if t["type"] != "PRICE":
                return rec
            self._last = self.epochTS(t["time"])

        if t["type"] == "PRICE":
            self.ask.append(float(t['closeoutAsk']))
            self.bid.append(float(t['closeoutBid']))
            self.ask_data["v"] += 1
            self.bid_data["v"] += 1

        if self.epochTS(t["time"]) - self._last >= self.interval:
            self.ask_data['o'] = self.ask[0]
            self.ask_data['h'] = np.max(self.ask)
            self.ask_data['l'] = np.min(self.ask)
            self.ask_data['c'] = self.ask[-1]
            self.ask = []
            self.bid_data['o'] = self.bid[0]
            self.bid_data['h'] = np.max(self.bid)
            self.bid_data['l'] = np.min(self.bid)
            self.bid_data['c'] = self.bid[-1]
            self.bid = []
            self._last = self.epochTS(t["time"])

            rec = (self.secs2time(self._last), self.ask_data['o'],
                   self.ask_data['h'], self.ask_data['l'], self.ask_data['c'], self.ask_data['v'],
                   self.bid_data['o'],
                   self.bid_data['h'], self.bid_data['l'], self.bid_data['c'], self.bid_data['v'])

            self.ask_data["v"] = 0
            self.bid_data["v"] = 0
            return rec
        return

    def granularity_to_time(self, gran):
        mfact = {'S': 1, 'M': 60, 'H': 3600, 'D': 86400}
        try:
            f, n = re.match(r"(?P<f>[SMHD])(?:(?P<n>\d+)|)",
                            gran).groups()
        except:
            raise ValueError("Can't handle granularity: {}".format(gran))
        else:
            n = int(n) if n else 1
            return mfact[f] * n

    def epochTS(self, t):
        d = datetime.strptime(t.split(".")[0], '%Y-%m-%dT%H:%M:%S')
        return int(calendar.timegm(d.timetuple()))

    def secs2time(self, e):
        w = time.gmtime(e)
        return datetime(*list(w)[0:6]).strftime("%Y-%m-%dT%H:%M:%S.000000Z")

class BotTrader(object):

    def __init__(self, instrument, granularity, units, clargs):
        self.accountID, token = exampleAuth()
        self.client = API(access_token=token)
        self.units = units
        self.clargs = clargs
        self.pt = PriceTable(instrument, granularity)
        mavgX = MAx(self.pt, clargs.shortMA, clargs.longMA, clargs.SOK, clargs.SOD, clargs.SOF)
        self.pt.setHandler("onAddItem", mavgX.calculate)
        self.indicators = [mavgX]
        self.state = NEUTRAL   # overall state based on calculated indicators

        # fetch initial historical data
        params = {"granularity": granularity,
                  "count": self.clargs.longMA}
        r = instruments.InstrumentsCandles(instrument=instrument,
                                           params=params)
        rv = self.client.request(r)
        # and calculate indicators
        for crecord in rv['candles']:
            if crecord['complete'] is True:
                self.pt.addItem(crecord['time'],
                                float(crecord['mid']['c']),
                                int(crecord['volume']),
                                float(crecord['mid']['o']),
                                float(crecord['mid']['h']),
                                float(crecord['mid']['l']))

        self._botstate()

    def _botstate(self):
        # overall state, in this case the state of the only indicator ...
        prev = self.state
        self.state = self.indicators[0].state
        units = self.units
        print('turned off orders ', sys._getframe().f_lineno)
        if self.state != prev and self.state in [SHORT, LONG]:
        #     logger.info("state change: from %s to %s", mapstate(prev),
        #                 mapstate(self.state))
            print('state change ', sys._getframe().f_lineno)
            print("state change: from %s to %s" % (mapstate(prev), mapstate(self.state)))
            # exit()
        #     units *= (1 if self.state == LONG else -1)
        #     self.close()
        #     self.order(units)

    def order(self, units):
        mop = {"instrument": self.pt.instrument,
               "units": units}

        def frmt(v):
            # format a number over 6 digits: 12004.1, 1.05455
            l = len(str(v).split(".")[0])
            return "{{:{}.{}f}}".format(l, 6-l).format(v)

        direction = 1 if units > 0 else -1
        if self.clargs.takeProfit:   # takeProfit specified? add it
            tpPrice = self.pt._c[self.pt.idx-1] * \
                      (1.0 + (self.clargs.takeProfit/100.0) * direction)
            mop.update({"takeProfitOnFill":
                        TakeProfitDetails(price=frmt(tpPrice)).data})

        if self.clargs.stopLoss:     # stopLosss specified? add it
            slPrice = self.pt._c[self.pt.idx-1] * \
                      (1.0 + (self.clargs.stopLoss/100.0) * -direction)
            mop.update({"stopLossOnFill":
                        StopLossDetails(price=frmt(slPrice)).data})

        data = MarketOrderRequest(**mop).data
        r = orders.OrderCreate(accountID=self.accountID, data=data)
        try:
            response = self.client.request(r)
        except V20Error as e:
            logger.error("V20Error: %s", e)
        else:
            logger.info("Response: %d %s", r.status_code,
                        json.dumps(response, indent=2))

    def close(self):
        logger.info("Close existing positions ...")
        r = positions.PositionDetails(accountID=self.accountID,
                                      instrument=self.pt.instrument)

        try:
            openPos = self.client.request(r)

        except V20Error as e:
            logger.error("V20Error: %s", e)

        else:
            toClose = {}
            for P in ["long", "short"]:
                if openPos["position"][P]["units"] != "0":
                    toClose.update({"{}Units".format(P): "ALL"})

            logger.info("prepare to close: {}".format(json.dumps(toClose)))
            r = positions.PositionClose(accountID=self.accountID,
                                        instrument=self.pt.instrument,
                                        data=toClose)
            rv = None
            try:
                if toClose:
                    rv = self.client.request(r)
                    logger.info("close: response: %s",
                                json.dumps(rv, indent=2))

            except V20Error as e:
                logger.error("V20Error: %s", e)

    def run(self):
        cf = PRecordFactory(self.pt.granularity)
        r = pricing.PricingStream(accountID=self.accountID,
                                  params={"instruments": self.pt.instrument})
        for tick in self.client.request(r):

            if 'PRICE' in tick['type']:
                rec = cf.parseTick(tick)

            if rec:
                self.pt.addItem(*rec)

            self._botstate()


# ------------------------
if __name__ == "__main__":

    granularities = defs.CandlestickGranularity().definitions.keys()
    print("choices: ", granularities)
    # exit()
    # create the top-level parser
    parser = argparse.ArgumentParser(prog='simplebot')
    parser.add_argument('--longMA', default=10, type=int,
                        help='period of the long movingaverage')
    parser.add_argument('--shortMA', default=3, type=int,
                        help='period of the short movingaverage')
    parser.add_argument('--stopLoss', default=20, type=float,
                        help='stop loss value as a percentage of entryvalue')
    parser.add_argument('--takeProfit', default=20, type=float,
                        help='take profit value as a percentage of entryvalue')
    parser.add_argument('--instrument', default='EUR_USD', type=str, help='instrument')
    parser.add_argument('--granularity', default='M1', choices=granularities)
    parser.add_argument('--units', default=1000000, type=int)
    parser.add_argument('--SOK', default=14, type=int,
                        help='period of the fast SO')
    parser.add_argument('--SOD', default=3, type=int,
                        help='period of the slow SO')
    parser.add_argument('--SOF', default=3, type=int,
                        help='period of the full SO')

    clargs = parser.parse_args()
    bot = BotTrader(instrument=clargs.instrument,
                    granularity=clargs.granularity,
                    units=clargs.units, clargs=clargs)
    bot.run()
