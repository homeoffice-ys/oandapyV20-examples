# https://admiralmarkets.com/education/articles/forex-strategy/forex-1-minute-scalping-strategy-explained
# -*- coding: utf-8 -*-
import re
import time
import argparse
from datetime import datetime
import calendar
import json
import logging
import pandas as pd
import numpy as np
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

from oandapyV20.definitions.instruments import CandlestickGranularity
from exampleauth import exampleAuth

from pyti.exponential_moving_average import exponential_moving_average as ema


""" Simple trading application based on MovingAverage crossover.

    Positions can be tracked using the oanda_console demo program.

    **********************************************************
    * THIS PROGRAM IS SOLELY MENT FOR DEMONSTRATION PURPOSE! *
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
    filename="./scalpbot.log",
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
    """Scalp via L&S Exp. MA and Full Stochastic Oscillator"""

    def __init__(self, pt, semaPeriod, lemaPeriod, FSO1Period, FSO2Period, FSO3Period):
        super(MAx, self).__init__(pt)
        self.semaPeriod = semaPeriod
        self.lemaPeriod = lemaPeriod
        self.FSO1Period = FSO1Period
        self.FSO2Period = FSO2Period
        self.FSO3Period = FSO3Period
        self._events = Event()
        self.state = NEUTRAL

    def calculate(self, idx):

        if idx <= self.lemaPeriod:   # not enough values to calculate leMAx
            self.values[idx-1] = None
            return

        # values = np.array(values)
        # pd.ewma(values, span=period)[-1]
        values = np.array(self._pt._c[idx-self.semaPeriod:idx])
        SEMA = pd.ewma(values, span=self.semaPeriod)[-1]
        values = np.array(self._pt._c[idx - self.lemaPeriod:idx])
        LEMA = pd.ewma(values, span=self.lemaPeriod)[-1]
        values = np.array(self._pt._h[idx - self.FSO1Period:idx])
        hh = values.max()
        values = np.array(self._pt._l[idx - self.FSO1Period:idx])
        ll = values.min()
        lso = 100 * np.asarray((self._pt._c[idx - self.FSO1Period:idx] - ll)/(hh - ll))
        # K = 100(C – LL) / (HH – LL)
        sso = sum(lso[0:self.FSO2Period]) / self.FSO2Period
        fso = sum(sso[0:self.FSO3Period]) / self.FSO3Period
        if SEMA > LEMA and fso < 20 and self._pt._c[idx] > SEMA:
            self.state = LONG
        elif SEMA < LEMA and fso > 80 and self._pt._c[idx] > LEMA:
            self.state = SHORT
        # self.values[idx-1] = SMA - LMA
        # self.state = LONG if self.values[idx-1] > 0 else SHORT
        logger.info("MAx: processed %s : state: %s",
                    self._pt[-1][0], mapstate(self.state))


class PriceTable(object):

    __slots__ = ['_dt', '_c', '_h', '_l', '_v', 'instrument', 'granularity', '_events', 'idx']

    def __init__(self, instrument, granularity):
        self.instrument = instrument
        self.granularity = granularity
        self._dt = [None] * 1000  # allocate space for datetime
        self._c = [None] * 1000   # allocate space for close values
        self._h = [None] * 1000  # allocate space for high values
        self._l = [None] * 1000  # allocate space for low values
        self._v = [None] * 1000   # allocate space for volume values
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

    def addItem(self, dt, c, h, l, v):
        self._dt[self.idx] = dt
        self._c[self.idx] = c
        self._h[self.idx] = h
        self._l[self.idx] = l
        self._v[self.idx] = v
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
            return (self._dt[_i], self._c[_i], self._h[_i], self._l[_i], self._v[_i])

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
        self.data = {"coa": None, "cob": None, "v": 0}

    def parseTick(self, t):
        rec = None
        if not self._last:
            if t["type"] != "PRICE":
                return rec
            epoch = self.epochTS(t["time"])
            self._last = epoch - (epoch % self.interval)

        if self.epochTS(t["time"]) > self._last + self.interval:
            # save this record as completed
            rec = (self.secs2time(self._last), self.data['c'], self.data['h'], self.data['l'], self.data['v'])
            # init new one
            self._last += self.interval
            self.data["v"] = 0

        if t["type"] == "PRICE":
            self.data["c"] = (float(t['closeoutBid']) +
                              float(t['closeoutAsk'])) / 2.0
            self.data["v"] += 1

        return rec

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
        mavgX = MAx(self.pt, clargs.shortEMA, clargs.longEMA, clargs.FSO1, clargs.FSO2, clargs.FSO3)
        self.pt.setHandler("onAddItem", mavgX.calculate)
        self.indicators = [mavgX]
        self.state = NEUTRAL   # overall state based on calculated indicators

        # fetch initial historical data
        params = {"granularity": granularity,
                  "count": self.clargs.longEMA}
        r = instruments.InstrumentsCandles(instrument=instrument,
                                           params=params)
        rv = self.client.request(r)
        # and calculate indicators
        for crecord in rv['candles']:
            if crecord['complete'] is True:
                self.pt.addItem(crecord['time'],
                                float(crecord['mid']['c']),
                                float(crecord['mid']['h']),
                                float(crecord['mid']['l']),
                                int(crecord['volume']))

        self._botstate()

    def _botstate(self):
        # overall state, in this case the state of the only indicator ...
        prev = self.state
        self.state = self.indicators[0].state
        units = self.units
        if self.state != prev and self.state in [SHORT, LONG]:
            logger.info("state change: from %s to %s", mapstate(prev),
                        mapstate(self.state))
            units *= (1 if self.state == LONG else -1)
            self.close()
            self.order(units)

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
            rec = cf.parseTick(tick)
            if rec:
                self.pt.addItem(*rec)

            self._botstate()


# ------------------------
if __name__ == "__main__":

    granularities = CandlestickGranularity().definitions.keys()
    # create the top-level parser
    parser = argparse.ArgumentParser(prog='simplebot')
    parser.add_argument('--shortEMA', default=50, type=int,
                        help='period of the exp. short movingaverage')
    parser.add_argument('--longEMA', default=100, type=int,
                        help='period of the exp. long  movingaverage')
    parser.add_argument('--FSO1', default=5, type=int,
                        help='period of the Full Stochastic Oscillator 1')
    parser.add_argument('--FSO2', default=3, type=int,
                        help='period of the Full Stochastic Oscillator 2')
    parser.add_argument('--FSO3', default=3, type=int,
                        help='period of the Full Stochastic Oscillator 3')
    parser.add_argument('--shortMA', default=2, type=int,
                        help='period of the short movingaverage')
    parser.add_argument('--stopLoss', default=1.5, type=float,
                        help='stop loss value as a percentage of entryvalue')
    parser.add_argument('--takeProfit', default=0.5, type=float,
                        help='take profit value as a percentage of entryvalue')
    parser.add_argument('--instrument', type=str, help='instrument', required=True)
    parser.add_argument('--granularity', choices=granularities, required=True)
    parser.add_argument('--units', type=int, required=True)

    clargs = parser.parse_args()
    bot = BotTrader(instrument=clargs.instrument,
                    granularity=clargs.granularity,
                    units=clargs.units, clargs=clargs)
    bot.run()
