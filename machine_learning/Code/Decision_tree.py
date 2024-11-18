import talib
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor

all = pd.read_csv(r'E:\02_Document\08_Python_Exercise\machine_learning\Result\close_price_data.csv',
                           index_col='Date').dropna()

features = all[['US Dollar Index Close Price']]
targets = all['VIX Close Price']

train_size = int(0.85 * targets.shape[0])
linear_features = sm.add_constant(features)
train_features = linear_features[:train_size]
train_targets = targets[:train_size]
test_features = linear_features[train_size:]
test_targets = targets[train_size:]

decision_tree = DecisionTreeRegressor(max_depth=5)
decision_tree.fit(train_features, train_targets)
print(decision_tree.score(train_features, train_targets))
print(decision_tree.score(test_features, test_targets))

train_predictions = decision_tree.predict(train_features)
test_predictions = decision_tree.predict(test_features)

plt.figure(figsize=(8, 6))
plt.scatter(train_targets, train_predictions, alpha=0.5, color='blue', label='Train')
plt.scatter(test_targets, test_predictions, alpha=0.5, color='red', label='Test')
xmin, xmax = plt.xlim()
plt.plot([xmin, xmax], [xmin, xmax], 'k--', label='Perfect Prediction')
plt.title('Decision Tree Regressor: Predictions vs Actuals')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.grid(alpha=0.3)
plt.show()