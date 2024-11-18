import talib
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

all = pd.read_csv(r'E:\02_Document\08_Python_Exercise\machine_learning\Result\close_price_data.csv',
                           index_col='Date').dropna()

# OLS US Dollar Index and VIX ——————————————————————————————————————————————————————————————————————————————————————————
features = all['US Dollar Index Close Price']
targets = all['VIX Close Price']
features.info()
targets.info()

linear_features = sm.add_constant(features)
linear_features.info()

train_size = int(0.85 * targets.shape[0])
train_features = linear_features[:train_size]
train_targets = targets[:train_size]
test_features = linear_features[train_size:]
test_targets = targets[train_size:]

model = sm.OLS(train_targets, train_features)
results = model.fit()
print(results.summary())
print(results.pvalues)

train_predictions = results.predict(train_features)
test_predictions = results.predict(test_features)

plt.scatter(train_targets, train_predictions, alpha=0.2, color='b', label='Train Predictions')
plt.scatter(test_targets, test_predictions, alpha=0.2, color='r', label='Test Predictions')

ymin, ymax = plt.ylim()
plt.plot([ymin, ymax], [ymin, ymax], 'k--', label='Perfect Prediction')

plt.title('Predictions vs Actuals (Train and Test)')
plt.xlabel('Actual')
plt.ylabel('Predictions')
plt.legend()  # 显示图例
plt.grid(alpha=0.3)  # 添加网格线
plt.show()