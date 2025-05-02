# 📈 Pairs Trading Algorithm 
This repository contains the Python code used in my undergraduate dissertation: \
**"Testing the Efficient Market Hypothesis in the Context of Algorithmic Trading: A Quantitative Analysis."**

The project implements a pairs trading strategy designed to exploit temporary mispricings between historically correlated assets, testing the persistence of weak-form market inefficiencies across different sectors and time periods.

---

## Features
- **Correlation filtering** of potential stock pairs
- **Cointegration testing**: Engel Granger and Johansen
- **Dynamic Hedge Ratio Calculation**: Kalman Filter
- **Backtesting** over changeable time horizon
- **Potfolio Construction** using Sharpe Ratios
- **Performance Metrics Calculation**: Including Calmar and Sortino Ratios
- **Econometrics tests**: Variance Ratios and Autocorrelation tests

---

## How to use
1. Set start and end date of backtest period
2. Establish stock tickers
3. Set threshold_1 over which stock pairs will be correlated
4. Determine whether to incorporate transaction costs - and if so set > 0
5. Set threshold_2: The Sharpe ratio benchmark over which pairs will be included in the constucted portfolio
6. Change (if you would like) the benchmark composite index stock ticker - from '^IXIC' to '^GSPC' for example
7. Run!

---
