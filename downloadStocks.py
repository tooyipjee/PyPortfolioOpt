import pandas as pd
import yfinance as yf
import datetime as dt
import numpy as np

today = dt.date.today() - dt.timedelta(days = 1)
start = today - dt.timedelta(days = 1 * 365)

stocks = pd.read_csv("stocks.csv")

timeseries_data = pd.DataFrame()
for i, symbol in enumerate(stocks.Symbol):
    print(i*100/len(stocks))
    data = yf.download(symbol, start=start, end=today)
    # days with bad data
    bad_days = data[data.Close == 0].index
    for bad_day in bad_days:
        avg_close_price = (data.loc[bad_day - dt.timedelta(days = 5):bad_day + dt.timedelta(days = 5)].Close)
        avg_close_price = np.mean(avg_close_price)               
        data.at[bad_day,'Close'] = avg_close_price

    timeseries_data = pd.concat([timeseries_data, pd.DataFrame({symbol: data.Close})], axis=1)

timeseries_data = timeseries_data.dropna(axis = 1, how = 'all')
timeseries_data.to_csv('stocks_prices.csv')