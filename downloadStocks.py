import pandas as pd
import yfinance as yf
import datetime as dt
import numpy as np
import csv 
from pypfopt import black_litterman
import os
    
symbols = 'penny_stocks'
years = 1

cwd = os.getcwd()
# define the name of the directory to be created
path = os.path.join(cwd, "outputs\\" + symbols)
try:
    os.mkdir(path)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s " % path)

today = dt.date.today() - dt.timedelta(days = 1)
start = today - dt.timedelta(days = years * 365)

stocks = pd.read_csv("C:/Users/Jason/Documents/GitHub/PyPortfolioOpt/symbols/"+ symbols +".csv")

timeseries_data = pd.DataFrame()
mcap_data = pd.DataFrame()
test = []
for i, symbol in enumerate(stocks.Symbol):
    print(i*100/len(stocks))
    data = yf.download(symbol, start=start, end=today)
    # days with bad data
    bad_days = data[data.Close == 0].index
    
    for bad_day in bad_days:
        avg_close_price = (data.loc[bad_day - dt.timedelta(days = 5):bad_day + dt.timedelta(days = 5)].Close)
        avg_close_price = np.mean(avg_close_price)               
        data.at[bad_day,'Close'] = avg_close_price
    mcap = data["Close"][-2] * data["Volume"][-2]
    delta = black_litterman.market_implied_risk_aversion(data['Close'])
    if (np.max(data.Close)/np.min(data.Close) < 20):
        timeseries_data = pd.concat([timeseries_data, pd.DataFrame({symbol: data.Close})], axis=1)
        test.append([symbol,mcap,delta])
        print(symbol + " passed")
    else:
        print(symbol + " failed")

timeseries_data = timeseries_data.dropna(axis = 1, how = 'all')
timeseries_data.to_csv(os.path.join(path,'stock_prices.csv'))

with open(os.path.join(path,'mcap.csv'), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(test)
    
