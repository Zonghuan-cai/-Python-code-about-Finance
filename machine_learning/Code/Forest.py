import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid
from pprint import pprint

all = pd.read_csv(
    r'E:\02_Document\08_Python_Exercise\machine_learning\Result\close_price_data.csv',
    index_col='Date'
).dropna()

features = all[['US Dollar Index Close Price']]
targets = all['VIX Close Price']

train_size = int(0.85 * targets.shape[0])
train_features = features[:train_size]
train_targets = targets[:train_size]
test_features = features[train_size:]
test_targets = targets[train_size:]

from sklearn.ensemble import RandomForestRegressor
random_forest = RandomForestRegressor(n_estimators=200, max_depth=5, max_features=4)
random_forest.fit(train_features, train_targets)

train_predictions = random_forest.predict(train_features)
test_predictions = random_forest.predict(test_features)

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