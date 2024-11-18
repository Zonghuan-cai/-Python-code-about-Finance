from pandas_datareader.data import DataReader
from datetime import date
import yfinance as yf
import pandas as pd

# Download —————————————————————————————————————————————————————————————————————————————————————————————————————————————
oil = yf.download('CL=F', start="2000-01-01", end="2024-11-18")
gold = yf.download('GC=F', start="2000-01-01", end="2024-11-18")
usd = yf.download('DX-Y.NYB', start="2000-01-01", end="2024-11-18")
vix = yf.download('^VIX', start="2000-01-01", end="2024-11-18")

close_price = pd.concat([oil['Close'], gold['Close'], usd['Close'], vix['Close']], axis=1).dropna()
close_price.columns = ['Oil Close Price', 'Gold Close Price', 'US Dollar Index Close Price', 'VIX Close Price']
close_price.to_csv(r'E:\02_Document\08_Python_Exercise\machine_learning\Result\close_price_data.csv',
                   index=True)

# Plot —————————————————————————————————————————————————————————————————————————————————————————————————————————————————
import matplotlib.pyplot as plt
fig, axs = plt.subplots(4, 1, figsize=(12, 18), sharex=True)

close_price['Oil Close Price'].plot(ax=axs[0], color='blue', linewidth=2, label='Oil')
axs[0].set_title('Oil Close Price')
axs[0].grid(True, linestyle='--', alpha=0.5)

close_price['Gold Close Price'].plot(ax=axs[1], color='green', linewidth=2, label='Gold')
axs[1].set_title('Gold Close Price')
axs[1].grid(True, linestyle='--', alpha=0.5)

close_price['US Dollar Index Close Price'].plot(ax=axs[2], color='red', linewidth=2, label='US Dollar Index')
axs[2].set_title('USD Close Price')
axs[2].grid(True, linestyle='--', alpha=0.5)

close_price['VIX Close Price'].plot(ax=axs[3], color='purple', linewidth=2, label='VIX')
axs[3].set_title('VIX Close Price')
axs[3].grid(True, linestyle='--', alpha=0.5)
plt.xlabel('Date', fontsize=12)
plt.show()

gold_vix_time_series = close_price[['Gold Close Price', 'VIX Close Price']].plot(
    figsize=(12, 6), secondary_y='VIX Close Price', title='Gold vs VIX Time Series'
)
gold_vix_time_series.right_ax.set_ylabel('VIX Close Price')
gold_vix_time_series.set_ylabel('Gold Close Price')
plt.xlabel('Date', fontsize=12)
plt.show()

# Corr —————————————————————————————————————————————————————————————————————————————————————————————————————————————————
import seaborn as sns
import statsmodels as sm
sns.heatmap(close_price.corr(), annot=True)
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

