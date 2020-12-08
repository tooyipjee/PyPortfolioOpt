import pandas as pd
import numpy as np
import cvxpy as cp
import os
import yfinance as yf
import datetime as dt
import csv
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import EfficientFrontier
from pypfopt import objective_functions
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt import HRPOpt
from pypfopt import CLA
from pypfopt import black_litterman
from pypfopt import BlackLittermanModel
from pypfopt import plotting
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices


def optimise_portfolio(stock_price, Portfolio):
    returns = stock_price.pct_change().dropna()
    # Calculate expected returns and sample covariance
    mu = expected_returns.mean_historical_return(stock_price)
    S = risk_models.sample_cov(stock_price)
    
    # Optimise for maximal Sharpe ratio
    ef = EfficientFrontier(mu, S)
    raw_weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    ef.save_weights_to_file("weights.csv")  # saves to file
    print(cleaned_weights)
    ef.portfolio_performance(verbose=True)
    Portfolio.ef = ef
    Portfolio.mu = mu
    Portfolio.S = S
    Portfolio.weights = cleaned_weights
    
def calculate_allocation(stock_price, cash, Portfolio):
    mu = expected_returns.mean_historical_return(stock_price)
    S = risk_models.sample_cov(stock_price)
    ef = EfficientFrontier(mu, S)
    cleaned_weights = ef.clean_weights()
    latest_prices = get_latest_prices(stock_price)
    da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=cash)
    allocation, leftover = da.lp_portfolio()
    print("Discrete allocation:", allocation)
    print("Funds remaining: ${:.2f}".format(leftover))
    
def plot_stock_insights(stock_price, Portfolio):
    mu = expected_returns.mean_historical_return(stock_price)
    S = risk_models.sample_cov(stock_price)
    ef = EfficientFrontier(mu, S)
    cleaned_weights = ef.clean_weights()
    cla = CLA(mu, S)
    plotting.plot_efficient_frontier(cla)
    plotting.plot_covariance(S)
    plotting.plot_weights(cleaned_weights)
    
def rebalance_weight(Portfolio):
    current_stocks = [i[3] for i in Portfolio.symbol_stats]
    current_weights = current_stocks/np.sum(current_stocks)
    change_needed_weights = current_weights - np.array(list(Portfolio.weights.values()))
    change_needed_stocks = change_needed_weights * current_stocks
    print(change_needed_stocks)
    
class Portfolio:
    i = 1
    def f(self):
        return 'Portfolio created'