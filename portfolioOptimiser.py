import pandas as pd
import numpy as np
import cvxpy as cp
import os
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

path = 'C:\\Users\\Jason\\Documents\\GitHub\\PyPortfolioOpt\\outputs\\penny_stocks'
# Reading in the data; preparing expected returns and a risk model
# df = pd.read_csv("tests/resources/stock_prices.csv", parse_dates=True, index_col="date")
df = pd.read_csv(os.path.join(path,'stock_prices.csv'), parse_dates=True, index_col="Date")

returns = df.pct_change().dropna()
# Calculate expected returns and sample covariance
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

# Optimise for maximal Sharpe ratio
ef = EfficientFrontier(mu, S)
raw_weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
ef.save_weights_to_file("weights.csv")  # saves to file
print(cleaned_weights)
ef.portfolio_performance(verbose=True)

latest_prices = get_latest_prices(df)

da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=600)
allocation, leftover = da.lp_portfolio()
print("Discrete allocation:", allocation)
print("Funds remaining: ${:.2f}".format(leftover))
cla = CLA(mu, S)
plotting.plot_efficient_frontier(cla)
plotting.plot_covariance(S)
plotting.plot_weights(cleaned_weights)
