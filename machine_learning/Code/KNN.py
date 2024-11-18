import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import statsmodels.api as sm

all = pd.read_csv(
    r'E:\02_Document\08_Python_Exercise\machine_learning\Result\close_price_data.csv',
    index_col='Date'
).dropna()

features = all[['US Dollar Index Close Price']]
targets = all['VIX Close Price']

train_size = int(0.85 * targets.shape[0])
linear_features = sm.add_constant(features)  # 添加常数项
train_features = linear_features[:train_size]
train_targets = targets[:train_size]
test_features = linear_features[train_size:]
test_targets = targets[train_size:]

scaler = StandardScaler()
scaled_train_features = scaler.fit_transform(train_features)
scaled_test_features = scaler.transform(test_features)

f, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
train_features.iloc[:, 1].hist(ax=ax[0], bins=20, alpha=0.7)
ax[0].set_title("Original Data Distribution")
ax[1].hist(scaled_train_features[:, 1], bins=20, alpha=0.7, color='orange')
ax[1].set_title("Scaled Data Distribution")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
for n in range(2, 13):
    knn = KNeighborsRegressor(n_neighbors=n)
    knn.fit(scaled_train_features, train_targets)

    print(f"n_neighbors = {n}")
    print(f"Train Score: {knn.score(scaled_train_features, train_targets):.4f}")
    print(f"Test Score: {knn.score(scaled_test_features, test_targets):.4f}")
    print()

    train_predictions = knn.predict(scaled_train_features)
    test_predictions = knn.predict(scaled_test_features)

    plt.scatter(train_predictions, train_targets, alpha=0.3, label=f'Train (k={n})')
    plt.scatter(test_predictions, test_targets, alpha=0.3, label=f'Test (k={n})')

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('KNN Regressor: Predictions vs Actuals')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
