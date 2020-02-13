# -*- coding: utf-8 -*-
"""Retrieve candle data.

For complete specs of the endpoint, please check:

    http://developer.oanda.com/rest-live-v20/instrument-ep/

Specs of InstrumentsCandles()

    http://oanda-api-v20.readthedocs.io/en/latest/oandapyV20.endpoints.html

"""
import argparse
import json
from oandapyV20 import API
from oandapyV20.exceptions import V20Error
import oandapyV20.endpoints.instruments as instruments
from oandapyV20.definitions.instruments import CandlestickGranularity
from exampleauth import exampleAuth
import re
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np
import os


# import plotly.plotly as py
# import plotly.graph_objs as go

# from sys import path as pylib #im naming it as pylib so that we won't get confused between os.path and sys.path
# import os
# pylib += [os.path.abspath(r'C:\Users\Yochanan\Documents\chaos')]

#import ssa_core
#import config

#from config import ssa_params

def save_to_jason_file(in_dict):
    dir_name = r'C:\Users\Yochanan\PycharmProjects\HMM\data\currency_data'
    suffix = '_test.json'
    with open(os.path.join(dir_name, in_dict['instrument'] + suffix), 'w') as json_file:
        json.dump(in_dict, json_file)

# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    train_size = int(len(X) * 0.66)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
        # calculate out of sample error
    error = mean_squared_error(test, predictions)
    return error


# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s MSE=%.7f' % (order,mse))
                except:
                    continue
    print('Best ARIMA%s MSE=%.7f' % (best_cfg, best_score))


price = ['M', 'B', 'A', 'BA', 'MBA']
granularities = CandlestickGranularity().definitions.keys()
# create the top-level parser
parser = argparse.ArgumentParser(prog='candle-data')
parser.add_argument('--nice', action='store_true', help='json indented')
parser.add_argument('--count', default=0, type=int,
                    help='num recs, if not specified 500')
parser.add_argument('--granularity', choices=granularities, required=True)
parser.add_argument('--price', choices=price, default='M', help='Mid/Bid/Ask')
parser.add_argument('--from', dest="From", type=str,
                    help="YYYY-MM-DDTHH:MM:SSZ (ex. 2016-01-01T00:00:00Z)")
parser.add_argument('--to', type=str,
                    help="YYYY-MM-DDTHH:MM:SSZ (ex. 2016-01-01T00:00:00Z)")
parser.add_argument('--instruments', type=str, nargs='?',
                    action='append', help='instruments')


class Main(object):
    def __init__(self, api, accountID, clargs):
        self._accountID = accountID
        self.clargs = clargs
        self.api = api

    def main(self):
        def check_date(s):
            dateFmt = "[\d]{4}-[\d]{2}-[\d]{2}T[\d]{2}:[\d]{2}:[\d]{2}Z"
            if not re.match(dateFmt, s):
                raise ValueError("Incorrect date format: ", s)

            return True

        if self.clargs.instruments:
            params = {}
            if self.clargs.granularity:
                params.update({"granularity": self.clargs.granularity})
            if self.clargs.count:
                params.update({"count": self.clargs.count})
            if self.clargs.From and check_date(self.clargs.From):
                params.update({"from": self.clargs.From})
            if self.clargs.to and check_date(self.clargs.to):
                params.update({"to": self.clargs.to})
            if self.clargs.price:
                params.update({"price": self.clargs.price})
            for i in self.clargs.instruments:
                r = instruments.InstrumentsCandles(instrument=i, params=params)
                rv = self.api.request(r)
                save_to_jason_file(rv)
                kw = {}
                if self.clargs.nice:
                    kw = {"indent": self.clargs.nice}
                print("{}".format(json.dumps(rv, **kw)))

                dath = [None] * len(rv['candles'])
                datl = [None] * len(rv['candles'])
                for idx, val in enumerate(rv['candles']):
                    dath[idx] = float(val['mid']['h'])
                    datl[idx] = float(val['mid']['l'])
                plt.plot(dath)
                plt.plot(datl)
                plt.title(rv['instrument'] + rv['granularity'])
                plt.show()

                # evaluate parameters
                p_values = [0, 1, 2, 4, 6, 8, 10]
                d_values = range(0, 3)
                q_values = range(0, 3)
                warnings.filterwarnings("ignore")
                evaluate_models(np.asarray(dath), p_values, d_values, q_values)




                # trace = go.Candlestick(x=rv['candles']['time'],
                #                        open=rv['candles']['mid']['o'],
                #                        high=rv['candles']['mid']['h'],
                #                        low=rv['candles']['mid']['l'],
                #                        close=rv['candles']['mid']['c'])
                #
                # layout = go.Layout(
                #     xaxis=dict(
                #         rangeslider=dict(
                #             visible=False
                #         )
                #     )
                # )
                #
                # data = [trace]
                #
                # fig = go.Figure(data=data, layout=layout)
                # py.iplot(fig, filename='simple_candlestick_without_range_slider')

if __name__ == "__main__":
    clargs = parser.parse_args()

    accountID, token = exampleAuth()
    api = API(access_token=token)
    try:
        m = Main(api=api, accountID=accountID, clargs=clargs)
        m.main()
    except V20Error as v20e:
        print("ERROR {} {}".format(v20e.code, v20e.msg))
    except ValueError as e:
        print("{}".format(e))
    except Exception as e:
        print("Unkown error: {}".format(e))
