import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

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

scaler = StandardScaler()
scaled_train_features = scaler.fit_transform(train_features)
scaled_test_features = scaler.transform(test_features)

train_targets = train_targets.values.reshape(-1, 1)
test_targets = test_targets.values.reshape(-1, 1)

model = Sequential()
model.add(Dense(50, input_dim=scaled_train_features.shape[1], activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mse')

history = model.fit(scaled_train_features, train_targets, epochs=50, verbose=0)

plt.plot(history.history['loss'])
plt.title('Loss: ' + str(round(history.history['loss'][-1], 6)))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

train_preds = model.predict(scaled_train_features).flatten()
test_preds = model.predict(scaled_test_features).flatten()
print("Train R²:", r2_score(train_targets, train_preds))
print("Test R²:", r2_score(test_targets, test_preds))

plt.figure(figsize=(8, 6))
plt.scatter(train_targets, train_preds, alpha=0.5, color='blue', label='Train')
plt.scatter(test_targets, test_preds, alpha=0.5, color='red', label='Test')
plt.plot(
    [min(train_targets), max(train_targets)],
    [min(train_targets), max(train_targets)],
    'k--', label='Perfect Prediction'
)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Neural Network: Predictions vs Actuals')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
