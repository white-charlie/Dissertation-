#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 12:47:39 2024

@author: charliewhite
"""

import yfinance as yf
import pandas as pd
from pykalman import KalmanFilter
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from matplotlib.ticker import PercentFormatter

colours = ['#2E86AB','#D1495B','#6A8EAE','#4A7C59','#C17817', '#5E4A8C','#3A3A3A']

# For yfinance
def set_data(tickers, start, end):
    # Download data with group_by='ticker'
    data = yf.download(tickers, start=start, end=end, group_by='ticker')
    
    # Extract 'Close' prices for each ticker
    close_prices = pd.DataFrame()
    for ticker in tickers:
        close_prices[ticker] = data[ticker]['Close']
    
    # Fill missing values
    close_prices = close_prices.fillna(method='ffill').fillna(method='bfill')
    
    return close_prices

start_date = '2020-01-01'
end_date = '2025-01-01'


tickers_tech = [
    'NVDA', 'AVGO', 'AMD', 'INTC', 'QCOM', 'ASML', 'TSM', 'MU', 'TXN', 'LRCX',
    'KLAC', 'AMAT', 'MRVL', 'NXPI', 'ON', 'MCHP', 'ADI', 'TER', 'SWKS', 'MPWR',
    'MSFT', 'CRM', 'NOW', 'SNOW', 'PANW', 'FTNT', 'CRWD', 'TEAM', 'DDOG', 'NET',
    'MDB', 'WDAY', 'ADBE', 'INTU', 'AMD',
    'REGN', 'VRTX', 'BIIB', 'ILMN', 'MRNA',
    'META', 'GOOGL', 'AMZN', 'BABA', 'MELI', 'SPOT', 'DOCU', 'ZS', 'TTD', 'ROKU'
]

tickers_commodities = [
    'TECK', 'VLO', 'NUE', 'MPC', 'FCX', 'GPK', 'LUN.TO', 'CAT', 'EMR', 'IYT', 
    'SLGN', 'FNV', 'XEL', 'HES', 'CVX', 'NXE', 'PKG', 'STLD', 'COP', 'RIO', 
    'BHP', 'SCCO', 'CCJ', 'COST', 'WPM', 'SHW', 'DE', 'CTVA', 'BG', 'AMKBY', 
    'WLK', 'SBLK', 'CF', 'APD', 'CS.TO', 'DOW', 'CSIQ', 'TAN', 'LPG', 
    'FRO', 'CTVA', 'BG', 'WLK', 'SBLK', 'CF', 'APD', 'CS.TO', 
    'DOW', 'CSIQ', 'TAN', 'LPG', 'FRO'
]

# Get stock data
data = set_data(tickers_commodities, start_date, end_date)

# Drop rows with missing values
data = data.dropna()

# Replace zero values with NaN and then forward-fill
data = data.replace(0, np.nan).ffill()

# Save to CSV and call
data.to_csv("stock_data.csv")
data = pd.read_csv("stock_data.csv", index_col=0, parse_dates=True)

# Compute correlation matrix
correlation_matrix = data.corr()

# Display correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5,annot_kws={'size': 5})
plt.title("Stock Correlation Heatmap")
plt.show()

#%%

#Test stock pairs for correlation

def get_high_correlation_pairs(corr_matrix, threshold):
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):  # Avoid duplicates
            if corr_matrix.iloc[i, j] >= threshold:
                stock1 = corr_matrix.columns[i]
                stock2 = corr_matrix.columns[j]
                correlation_value = corr_matrix.iloc[i, j]
                high_corr_pairs.append((stock1, stock2, correlation_value))
    
    return sorted(high_corr_pairs, key=lambda x: -x[2])  # Sort by highest correlation first

threshold_1 = 0.9

# Get stock pairs with correlation above threshold
high_correlation_stocks = get_high_correlation_pairs(correlation_matrix, threshold_1)

results = []
cumulative_returns_dict = {}

plt.figure(figsize=(12, 8))

# Display results
print(f"Highly Correlated Stock Pairs (Threshold: {threshold_1}):")


for stock1, stock2, corr in high_correlation_stocks:
    print(f"{stock1} - {stock2}: {corr:.2f}")
    
    # Engel-Granger and Johansen cointegration tests
    score, p_value, _ = sm.tsa.coint(data[stock1], data[stock2])
    johansen_result = coint_johansen(data[[stock1, stock2]], det_order=0, k_ar_diff=1)
    trace_statistic = johansen_result.lr1[0]
    critical_value_5 = johansen_result.cvt[0, 1]

    # Check if cointegrated
    if p_value < 0.05 or trace_statistic > critical_value_5:
        print(f"{stock1} and {stock2} are cointegrated. Running...")

        #Use Kalman Filter for dynamic hedge ratio estimation
        kf = KalmanFilter(transition_matrices=[1],
                          observation_matrices=[1],
                          initial_state_mean=0,
                          initial_state_covariance=1,
                          observation_covariance=0.1,
                          transition_covariance=0.001)

        state_means, _ = kf.filter(data[stock1].values.reshape(-1, 1))
        data['Hedge_Ratio'] = state_means.flatten()
        
        #Calculate Spread based on a rolling hedge ratio
        data['Spread'] = data[stock1] - data['Hedge_Ratio'] * data[stock2]

        # Z-Score & Trading Signals
        data['Rolling_Mean'] = data['Spread'].rolling(window=30).mean()
        data['Rolling_Std'] = data['Spread'].rolling(window=30).std()
        data['Z_Score'] = (data['Spread'] - data['Rolling_Mean']) / data['Rolling_Std']
        
        # Define entry and exit Z-scores
        upper_threshold, lower_threshold = 2, 0
        

        data['Position'] = 0
        data['long entry'] = ((data['Z_Score'] < -upper_threshold) & (data['Z_Score'].shift(1) > -upper_threshold))
        data['long exit'] = ((data['Z_Score'] > -lower_threshold) & (data['Z_Score'].shift(1) < -lower_threshold))
        
        # Set long position (1 for entry, 0 for exit)
        data['Position'] = np.where(data['long entry'], 1, data['Position'])
        data['Position'] = np.where(data['long exit'], 0, data['Position'])
        
        # Set up short entry and exit conditions
        data['short entry'] = ((data['Z_Score'] > upper_threshold) & (data['Z_Score'].shift(1) < upper_threshold))
        data['short exit'] = ((data['Z_Score'] < lower_threshold) & (data['Z_Score'].shift(1) > lower_threshold))
        
        # Set short position (-1 for entry, 0 for exit)
        data['Position'] = np.where(data['short entry'], -1, data['Position'])
        data['Position'] = np.where(data['short exit'], 0, data['Position'])

        # Shift position to avoid look-ahead bias
        data['Position'] = data['Position'].shift(1)
        
        # Calculate Returns

        # Initially model with zero transation costs
        transaction_cost = 0.000
        data['Spread_Return'] = data['Spread'].pct_change()
        data['Strategy_Return'] = data['Position'].shift(1) * data['Spread_Return']
        data['Strategy_Return'] -= transaction_cost * abs(data['Position'].diff())

        # Calculate Cumulative Return
        data['Cumulative_Return'] = (1 + data['Strategy_Return']).cumprod()
        pair_name = f"{stock1}-{stock2}"
        cumulative_returns_dict[pair_name] = data['Cumulative_Return'].copy()
        
        # Sharpe Ratio
        sharpe_ratio = data['Strategy_Return'].mean() / data['Strategy_Return'].std() * np.sqrt(252)
        
        # Max Drawdown
        cumulative_max = data['Cumulative_Return'].cummax()
        drawdown = (cumulative_max - data['Cumulative_Return']) / cumulative_max
        max_drawdown = drawdown.max()
        
        annualized_return = (data['Cumulative_Return'].iloc[-1] ** (252/len(data))) - 1 
        calmar_ratio = annualized_return / max_drawdown if max_drawdown != 0 else np.nan

        results.append((stock1, stock2, corr, sharpe_ratio, max_drawdown, data['Cumulative_Return'].iloc[-1], calmar_ratio))
        plt.plot(data.index, data['Cumulative_Return'], label=pair_name, linewidth=1.5)

    else:
        print(f"{stock1} and {stock2} are NOT cointegrated. Skipping.")
        
results_df = pd.DataFrame(results, columns=['Stock1', 'Stock2', 'Correlation', 'Sharpe Ratio', 'Max Drawdown', 'Final Return','Calmar Ratio'])
print("\nFinal Results:")
print(results_df.sort_values(by='Sharpe Ratio', ascending=False))

plt.rcParams.update({'font.size': 12})

plt.title("Cumulative Returns of Cointegrated Pairs Trading Strategies")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend(loc="upper left", fontsize="small", bbox_to_anchor=(1, 1))
plt.grid()
plt.show()


#%%

# Filter results to include only pairs with Sharpe ratios above threshold_2
threshold_2 = 0.9
filtered_results = [result for result in results if result[3] >= threshold_2] # result[3] is Sharpe ratio

#Plot Cumulative Returns for pairs with Sharpe ratios of 1 or above

plt.figure(figsize=(12, 8))

for result in filtered_results:
    stock1, stock2, corr, sharpe_ratio, max_drawdown, _, calmar_ratio = result
    pair_name = f"{stock1}-{stock2}"
    
    cumulative_return = cumulative_returns_dict[pair_name]
    cumulative_return = cumulative_return.reindex(data.index, method='ffill')
    
    plt.plot(data.index, cumulative_return, label=pair_name, linewidth=1.5)

plt.title(f"Cumulative Returns of Pairs with Sharpe Ratios >= {threshold_2}", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Cumulative Return", fontsize=12)
plt.legend(title="Stock Pairs", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid()
plt.show()

#%% Create portfolio from 'filtered_results', of pairs with Sharpe ratio > threshold_2


# Assume equal weight allocation 

portfolio_returns_p = pd.Series(0, index=data.index)

for result in filtered_results:
    stock1, stock2, corr, sharpe_ratio, max_drawdown, _, calmar_ratio = result
    pair_name = f"{stock1}-{stock2}"
    pair_returns = cumulative_returns_dict[pair_name].pct_change().fillna(0)
    portfolio_returns_p += pair_returns / len(filtered_results)
    
#Calculate Portfolio Performance Metrics

# Compute cumulative returns
portfolio_cumulative_returns_p = (1 + portfolio_returns_p).cumprod()
portfolio_cumulative_returns_p = portfolio_cumulative_returns_p / portfolio_cumulative_returns_p.iloc[0]

# Calculate drawdowns
cumulative_max_p = portfolio_cumulative_returns_p.cummax()
portfolio_drawdowns_p = (portfolio_cumulative_returns_p - cumulative_max_p) / cumulative_max_p
max_drawdown_p = portfolio_drawdowns_p.min()

# Annualized Return
annualized_return_p = (portfolio_cumulative_returns_p.iloc[-1] ** (252 / len(portfolio_returns_p))) - 1

# Annualized Volatility
annualized_vol_p = portfolio_returns_p.std() * np.sqrt(252)

# Sharpe Ratio (assuming risk-free rate = 0)
sharpe_ratio_p = annualized_return_p / annualized_vol_p if annualized_vol_p != 0 else np.nan

# Calmar Ratio
calmar_ratio_p = annualized_return_p / abs(max_drawdown_p) if max_drawdown_p != 0 else np.nan

# Cumulative Return
cumulative_return_p = portfolio_cumulative_returns_p.iloc[-1] - 1  # As percentage

# Sortino Ratio (focuses on downside volatility)
downside_returns = portfolio_returns_p[portfolio_returns_p < 0]
downside_vol = downside_returns.std() * np.sqrt(252)
sortino_ratio_p = annualized_return_p / downside_vol if downside_vol != 0 else np.nan

performance_metrics = {
    "Annualised Return": annualized_return_p,
    "Cumulative Return": cumulative_return_p,
    "Annualised Volatility": annualized_vol_p,
    "Max Drawdown": max_drawdown_p,
    "Sharpe Ratio": sharpe_ratio_p,
    "Calmar Ratio": calmar_ratio_p,
    "Sortino Ratio": sortino_ratio_p,
}

performance_df = pd.DataFrame([performance_metrics])
performance_df = performance_df.T.rename(columns={0: "Portfolio Performance"})
performance_df.style.format({
    "Portfolio Performance": "{:.2%}"  # Format as percentage
})

performance_df

#%% Plot Cum rets and drawdowns

# Plot 1: Cumulative Returns
plt.figure(figsize=(12, 6))
plt.plot(portfolio_cumulative_returns_p, 
         label='Pairs Strategy Portfolio', 
         color=colours[0],
         linewidth=2)
plt.title('Pairs Trading Strategy: Cumulative Returns', pad=10, fontsize=14)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Cumulative Returns', fontsize=14)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))  # Format as percentages
plt.legend(frameon=True, facecolor='white')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Plot 2: Drawdowns
plt.figure(figsize=(12, 6))
plt.plot(portfolio_drawdowns_p, 
         label='Drawdowns', 
         color=colours[1],
         linewidth=1.5)
plt.fill_between(portfolio_drawdowns_p.index, 
                 portfolio_drawdowns_p, 
                 color=colours[1], 
                 alpha=0.15)
plt.axhline(0, linestyle='--', color='black', linewidth=0.8, label='Baseline')
plt.title('Pairs Trading Strategy: Drawdown Analysis', pad=10, fontsize=14)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Drawdown', fontsize=14)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))  # Format as percentages
plt.legend(frameon=True, facecolor='white')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#%% Comparison to becnhmark composite index

# Alter benchmark if desired
benchmark = ['^IXIC']
nasdaq = yf.download(benchmark[0], start=start_date, end=end_date )

nasdaq['Returns'] = nasdaq['Close'].pct_change()
nasdaq['Cumulative Returns'] = (1 + nasdaq['Returns']).cumprod()

# Compute daily log returns
nasdaq["Daily_Return"] = np.log(nasdaq["Close"] / nasdaq["Close"].shift(1))
nasdaq["Sharpe Ratio"] = nasdaq["Daily_Return"].mean()/ nasdaq["Daily_Return"].std() * np.sqrt(252)

nasdaq = nasdaq.dropna()
nasdaq_annual_return = (nasdaq['Cumulative Returns'].iloc[-1] ** (252 / len(nasdaq))) - 1
nasdaq_cumulative_return = nasdaq['Cumulative Returns'].iloc[-1] - 1
nasdaq_volatility = nasdaq['Returns'].std() * np.sqrt(252)
nasdaq_max_drawdown = (nasdaq['Close'] / nasdaq['Close'].cummax() - 1).min()
nasdaq_max_drawdown = nasdaq_max_drawdown.item()
nasdaq_sharpe = nasdaq['Daily_Return'].mean() / nasdaq['Daily_Return'].std() * np.sqrt(252)
nasdaq_calmar = nasdaq_annual_return / abs((nasdaq_max_drawdown)) if nasdaq_max_drawdown != 0 else np.nan
nasdaq_sortino = nasdaq_annual_return / (nasdaq['Returns'][nasdaq['Returns'] < 0].std() * np.sqrt(252))

nasdaq_metrics = {
    "Annualised Return": nasdaq_annual_return,
    "Cumulative Return": nasdaq_cumulative_return,
    "Annualised Volatility": nasdaq_volatility,
    "Max Drawdown": nasdaq_max_drawdown,
    "Sharpe Ratio": nasdaq_sharpe,
    "Calmar Ratio": nasdaq_calmar,
    "Sortino Ratio": nasdaq_sortino,
}

nasdaq_df = pd.DataFrame([nasdaq_metrics])
nasdaq_df = nasdaq_df.T.rename(columns={0: "Nasdaq Performance"})
nasdaq_df.style.format({
    "Portfolio Performance": "{:.2%}"  # Format as percentage
})

nasdaq_df

plt.figure(figsize = (12,8))

plt.plot(nasdaq.index, nasdaq['Cumulative Returns'], label = f'{benchmark[0]} Composite Benchmark', color=colours[1], linestyle = 'dashed')
plt.plot(portfolio_cumulative_returns_p.index, portfolio_cumulative_returns_p, label = ' Portfolio Pairs strategy', color = colours[0])
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.title(f'Cumulative Returns: Pairs Trading vs {benchmark[0]} Composite')
plt.legend()
plt.show()

#%% 

# ------------------Perform Econometrics Tests------------------------

pair_names = []
cumulative_returns_list = []
buy_hold_cumulative_returns_dict = {}

for stock1, stock2, corr, sharpe_ratio, max_drawdown, final_return, calmar_ratio in filtered_results:
    pair_name = f"{stock1}-{stock2}"
    pair_names.append(pair_name)
    cumulative_returns_list.append(cumulative_returns_dict[pair_name])
    buy_hold_cumulative_return = ((data[stock1].pct_change() + data[stock2].pct_change()) / 2).add(1).cumprod()
    buy_hold_cumulative_returns_dict[pair_name] = buy_hold_cumulative_return

aligned_returns = pd.DataFrame(index=data.index)
for pair_name, returns in zip(pair_names, cumulative_returns_list):
    aligned_returns[pair_name] = returns
    
buy_hold_aligned_returns = pd.DataFrame(index=data.index)
for pair_name, returns in buy_hold_cumulative_returns_dict.items():
    buy_hold_aligned_returns[pair_name] = returns

aligned_returns = aligned_returns.fillna(1.0)
buy_hold_aligned_returns = buy_hold_aligned_returns.fillna(1.0)

# Calculate daily returns for each pair
pairs_daily_returns = aligned_returns.pct_change().dropna()
buy_hold_daily_returns = buy_hold_aligned_returns.pct_change().dropna()

# Compute daily returns for the pairs trading strategy
pairs_trading_returns = pairs_daily_returns.mean(axis=1)
buy_hold_returns = buy_hold_daily_returns.mean(axis=1)

#%% Variance Ratio

import statsmodels.api as sm
from scipy.stats import norm, rankdata

# Variance Ratio function in line with Wright (2000)
def variance_ratio_test(returns, k):
    
    T = len(returns)  # Sample size
    
    # Rank-based transformation 1
    ranks = rankdata(returns)  # Assign ranks to returns
    r1_t = (ranks - T + (1/2)) / np.sqrt((T - 1) * (T + 1) / 12)  # Standardized ranks
    rolling_r1 = pd.Series(r1_t).rolling(window=k).sum().dropna()
    
    # Variance of rank-based statistic 1
    var_r1_1 = np.var(r1_t, ddof=1)
    var_r1_k = np.var(rolling_r1, ddof=1)
    vr_r_1 = var_r1_k / (k * var_r1_1)
    
    # Calculate Rank-based test statistic 1
    phi_k = 2 * (2 * k - 1) * (k - 1) / (3 * k * T)
    R_1 = (vr_r_1 - 1) / np.sqrt(phi_k)
    p_R_1 = 2 * (1 - norm.cdf(abs(R_1)))  # Two-tailed test

    # Rank-based test statistic 2
    r2_t = norm.ppf(ranks / (T + 1))  # Inverse normal CDF of rank
    rolling_r2 = pd.Series(r2_t).rolling(window=k).sum().dropna()
    
    # Variance of Rank-based statistic 2
    var_r2_1 = np.var(r2_t, ddof=1)
    var_r2_k = np.var(rolling_r2, ddof=1)
    vr_r_2 = var_r2_k / (k * var_r2_1)
    
    # Calculate Rank-based test statistic 2
    R_2 = (vr_r_2 - 1) / np.sqrt(phi_k)
    p_R_2 = 2 * (1 - norm.cdf(abs(R_2)))

    return vr_r_1, R_1, p_R_1, vr_r_2, R_2, p_R_2

# Test Variance Ratio at various lags
T = len(pairs_trading_returns)
print(f"T={T}")

# Perform Variance Ratio Test
lags = [2, 3, 4, 5, 10, 20, 30, 40]
columns = ["Strategy", "Lag", "VR1", "R1", "p-value (R1)", "VR2", "R2", "p-value (R2)"]
results = []

# Run variance ratio tests for the pairs trading strategy
for k in lags:
    vr_r_1, R_1, p_R_1, vr_r_2, R_2, p_R_2 = variance_ratio_test(pairs_trading_returns, k)
    results.append(["Pairs Trading", k, vr_r_1, R_1, p_R_1, vr_r_2, R_2, p_R_2])

# Run variance ratio tests for the buy-and-hold portfolio
for k in lags:
    vr_r_1, R_1, p_R_1, vr_r_2, R_2, p_R_2 = variance_ratio_test(buy_hold_returns, k)
    results.append(["Buy-and-Hold", k, vr_r_1, R_1, p_R_1, vr_r_2, R_2, p_R_2])

df_results = pd.DataFrame(results, columns=columns)

df_pairs_trading = df_results[df_results["Strategy"] == "Pairs Trading"].drop(columns=["Strategy"])
df_buy_and_hold = df_results[df_results["Strategy"] == "Buy-and-Hold"].drop(columns=["Strategy"])

def format_df(df):
    df["VR1"] = df["VR1"].apply(lambda x: f"{x:.4f}")
    df["R1"] = df["R1"].apply(lambda x: f"{x:.4f}")
    df["p-value (R1)"] = df["p-value (R1)"].apply(lambda x: f"{x:.4f}")
    df["VR2"] = df["VR2"].apply(lambda x: f"{x:.4f}")
    df["R2"] = df["R2"].apply(lambda x: f"{x:.4f}")
    df["p-value (R2)"] = df["p-value (R2)"].apply(lambda x: f"{x:.4f}")
    return df

df_pairs_trading = format_df(df_pairs_trading)
df_buy_and_hold = format_df(df_buy_and_hold)

print("Pairs Trading Results:")
print(df_pairs_trading)

print("\nBuy-and-Hold Results:")
print(df_buy_and_hold)

def save_df_as_image(df, filename):
    fig, ax = plt.subplots(figsize=(12, 4))  # Adjust figure size as needed
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

# Save the tables as images
save_df_as_image(df_pairs_trading, "pairs_trading_results.png")
save_df_as_image(df_buy_and_hold, "buy_and_hold_results.png")

#%% Autocorrelation and Ljung-Box Q-test

from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox

# Define your fixed lags of interest
lags = [2, 3, 4, 5, 10, 20, 30, 40]

pairs_trading_returns = pd.Series(pairs_trading_returns)
buy_hold_returns = pd.Series(buy_hold_returns)

def run_autocorrelation_analysis(returns, strategy_name):
    """Helper function to run autocorrelation analysis"""
    print(f"\n=== {strategy_name} Strategy Autocorrelation Analysis ===")
    
    # Compute autocorrelation at all specified lags
    autocorr_values = acf(returns, nlags=max(lags), fft=False)
    
    # Ljung-Box Q-Test at specified lags
    lb_test = acorr_ljungbox(returns, lags=lags, return_df=True)
    
    print("\nLag | Autocorrelation | Ljung-Box Stat | p-value")
    print("-----------------------------------------------")
    for lag in lags:
        print(f"{lag:3d} | {autocorr_values[lag]:.5f}        | {lb_test.loc[lag, 'lb_stat']:10.3f} | {lb_test.loc[lag, 'lb_pvalue']:.5f}")
    
    # Plot autocorrelation function
    sm.graphics.tsa.plot_acf(returns, lags=max(lags), title=f'{strategy_name} Strategy ACF', alpha=0.05, bartlett_confint=True)
    plt.show()

# Run analysis for both strategies
run_autocorrelation_analysis(pairs_trading_returns, "Pairs Trading")
run_autocorrelation_analysis(buy_hold_returns, "Buy-and-Hold")

#%% Breusch-Godfrey (LM) autocorrelation test
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
from statsmodels.tsa.tsatools import lagmat
from scipy.stats import chi2

lm_test_results = {}

# Define number of lags for Breusch-Godfrey test
nlags = 10

for stock1, stock2, _, _, _, _, _ in filtered_results:
    
    # Define the dependent and independent variables
    y = data[stock1]  # Target asset
    X = pd.DataFrame({"x_t": data[stock2], "y_t-1": data[stock1].shift(1)})  # Include lagged y_t-1
    X = sm.add_constant(X.dropna())  # Add intercept and drop NaNs

    # Align y with X after dropping NaNs
    y = y.loc[X.index]  

    # Run the regression
    model = sm.OLS(y, X).fit()
    
    # Extract residuals
    residuals = model.resid

    # Create auxiliary regression variables: original regressors + lagged residuals
    lagged_residuals = lagmat(residuals, maxlag=nlags, trim='both')  # Generate lagged residuals
    aux_X = pd.DataFrame(lagged_residuals, index=residuals.index[nlags:], columns=[f"eps_lag{j+1}" for j in range(nlags)])
    aux_X = sm.add_constant(pd.concat([X.iloc[nlags:], aux_X], axis=1))  # Include original regressors

    # Align residuals with auxiliary regression variables
    aux_y = residuals.iloc[nlags:]

    # Run auxiliary regression
    aux_model = sm.OLS(aux_y, aux_X).fit()
    r_squared_aux = aux_model.rsquared  

    # Compute LM test statistic
    LM_stat = len(aux_y) * r_squared_aux  # T * R^2 from auxiliary regression
    p_value = chi2.sf(LM_stat, df=nlags)  # Chi-squared p-value

    lm_test_results[f"{stock1}-{stock2}"] = {"LM Statistic": LM_stat, "p-value": p_value}

print("\nBreusch-Godfrey LM Test Results:\n")
print("Stock Pair        | LM Statistic | p-value ")
print("--------------------------------------------")
for pair, results in lm_test_results.items():
    print(f"{pair:15} | {results['LM Statistic']:.4f}     | {results['p-value']:4f}")

