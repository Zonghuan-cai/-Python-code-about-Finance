import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor

tickers = ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "GOOG", "TSLA", "BRK-B", "JPM","^GSPC"]
data = yf.download(tickers, start="2012-05-18", end="2024-11-19")
data = data['Close']
data.to_csv(r'E:\02_Document\08_Python_Exercise\modern_portfolio_theory\Result\top10.csv',
                   index=True)

returns_daily = data.pct_change()
monthly_df = data.resample('BMS').first()
returns_monthly = monthly_df.pct_change().dropna()
print(returns_monthly.head())

convariances = {}
for i in returns_monthly.index:
    rtd_idx = returns_daily.index
    mask = (rtd_idx.month == i.month) & (rtd_idx.year == i.year)
    convariances[i] = returns_daily[mask].cov()

portfolio_returns, portfolio_volatility, portfolio_weights = {}, {}, {}
for date in convariances.keys():
    cov = convariances[date]
    for single_portfolio in range(5000):
        weights = np.random.random(11)
        weights = weights / np.sum(weights)
        returns = np.dot(weights, returns_monthly.loc[date])
        volatility = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        portfolio_returns.setdefault(date, []).append(returns)
        portfolio_volatility.setdefault(date, []).append(volatility)
        portfolio_weights.setdefault(date, []).append(weights)
date = sorted(convariances.keys())[-1]

plt.scatter(x = portfolio_returns[date],
            y = portfolio_volatility[date],
            alpha = 0.5)
plt.xlabel('Volatility')
plt.ylabel('Returns')
plt.show()

now = datetime.now().strftime("%Y-%m-%d")
daily_risk_free_rate = yf.download("^TNX", start=now, end=now, interval="1d")['Close']

# 检查是否有无风险利率数据
if daily_risk_free_rate.empty:
    daily_risk_free_rate_value = 0
else:
    daily_risk_free_rate_value = daily_risk_free_rate.iloc[0] / 100 / 252  # 转换为日化收益率

sharpe_ratio, max_sharpe_idxs = {}, {}
for date in portfolio_returns.keys():
    for i, ret in enumerate(portfolio_returns[date]):
        volatility = portfolio_volatility[date][i]
        if daily_risk_free_rate_value == 0:
            # 无风险利率为0时直接计算
            sharpe_ratio.setdefault(date, []).append(ret / volatility)
        else:
            # 否则使用完整公式
            sharpe_ratio.setdefault(date, []).append((ret - daily_risk_free_rate_value) / volatility)
    max_sharpe_idxs[date] = np.argmax(sharpe_ratio[date])


ewma_daily = returns_daily.ewm(span = 30).mean()
ewma_monthly = ewma_daily.resample('BMS').first()
ewma_monthly = ewma_monthly.shift(1).dropna()

targets, features = [], []
for date, ewma in ewma_monthly.iterrows():
    best_idx = max_sharpe_idxs[date]
    targets.append(portfolio_weights[date][best_idx])
    features.append(ewma)

targets = np.array(targets)
features = np.array(features)

date = sorted(convariances.keys())[-1]
cur_returns = portfolio_returns[date]
cur_volatility = portfolio_volatility[date]
plt.scatter(x=cur_volatility, y=cur_returns, alpha=0.5, label='Portfolios')
best_idx = max_sharpe_idxs[date]
plt.scatter(x=cur_volatility[best_idx], y=cur_returns[best_idx], color='red', label='Max Sharpe Portfolio')
plt.xlabel('Volatility')
plt.ylabel('Returns')
plt.legend()
plt.title('Portfolio Optimization')
plt.grid(True)
plt.show()

# Apply by Machine learning
train_size = int(0.8*features.shape[0])
train_features = features[:train_size]
train_targets = targets[:train_size]

test_features = features[train_size:]
test_targets = targets[train_size:]

rfr = RandomForestRegressor(n_estimators=300)
rfr.fit(train_features, train_targets)

test_predictions = rfr.predict(test_features)
test_returns = np.sum(returns_monthly.iloc[train_size:] * test_predictions, axis =1)

plt.plot(test_returns, label = 'model')
plt.plot(returns_monthly['^GSPC'].iloc[train_size:], label = 'S&P 500 Index')
plt.legend()
plt.show()

cash = 1000
model_cash = [cash]
for r in test_returns:
    cash *= 1 + r
    model_cash.append(cash)

cash = 1000
sp_500_Index_cash = [cash]
for r in returns_monthly['^GSPC'].iloc[train_size:]:
    cash *= 1 + r
    sp_500_Index_cash.append(cash)

print('model returns', (model_cash[-1] - model_cash[0])/model_cash[0])
print('S&P 500 Index returns', (sp_500_Index_cash[-1] - sp_500_Index_cash[0])/sp_500_Index_cash[0])

plt.plot(model_cash, label = 'model')
plt.plot(sp_500_Index_cash, label = 'S&P 500 Index')
plt.ylabel('USD')
plt.legend()
plt.show()