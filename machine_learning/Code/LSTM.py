import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

all = pd.read_csv(
    r'E:\02_Document\08_Python_Exercise\machine_learning\Result\close_price_data.csv',
    index_col='Date'
).dropna()

features = all[['US Dollar Index Close Price', 'Gold Close Price']]
vix_targets = all['VIX Close Price']
gold_targets = all['Gold Close Price']

scaler_features = MinMaxScaler()
scaler_vix_targets = MinMaxScaler()
scaler_gold_targets = MinMaxScaler()

scaled_features = scaler_features.fit_transform(features)
scaled_vix_targets = scaler_vix_targets.fit_transform(vix_targets.values.reshape(-1, 1))
scaled_gold_targets = scaler_gold_targets.fit_transform(gold_targets.values.reshape(-1, 1))

train_size = int(0.85 * scaled_vix_targets.shape[0])
train_features = scaled_features[:train_size]
train_vix_targets = scaled_vix_targets[:train_size]
train_gold_targets = scaled_gold_targets[:train_size]
test_features = scaled_features[train_size:]
test_vix_targets = scaled_vix_targets[train_size:]
test_gold_targets = scaled_gold_targets[train_size:]

def create_sequences(data, targets, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        label = targets[i+seq_length]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

seq_length = 10
train_X, train_vix_y = create_sequences(train_features, train_vix_targets, seq_length)
train_gold_y = train_gold_targets[seq_length:]
test_X, test_vix_y = create_sequences(test_features, test_vix_targets, seq_length)
test_gold_y = test_gold_targets[seq_length:]

train_X = torch.from_numpy(train_X).float()
train_vix_y = torch.from_numpy(train_vix_y).float()
train_gold_y = torch.from_numpy(train_gold_y).float()
test_X = torch.from_numpy(test_X).float()
test_vix_y = torch.from_numpy(test_vix_y).float()
test_gold_y = torch.from_numpy(test_gold_y).float()

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

vix_model = LSTMModel(input_dim=train_X.shape[2], hidden_dim=50, output_dim=1)
gold_model = LSTMModel(input_dim=train_X.shape[2], hidden_dim=50, output_dim=1)

criterion = nn.MSELoss()
vix_optimizer = torch.optim.Adam(vix_model.parameters(), lr=0.001)
gold_optimizer = torch.optim.Adam(gold_model.parameters(), lr=0.001)

for epoch in range(100):
    vix_model.train()
    vix_optimizer.zero_grad()
    vix_outputs = vix_model(train_X)
    vix_loss = criterion(vix_outputs, train_vix_y.view(-1, 1))
    vix_loss.backward()
    vix_optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}, VIX Loss: {vix_loss.item()}')

for epoch in range(100):
    gold_model.train()
    gold_optimizer.zero_grad()
    gold_outputs = gold_model(train_X)
    gold_loss = criterion(gold_outputs, train_gold_y.view(-1, 1))
    gold_loss.backward()
    gold_optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}, Gold Loss: {gold_loss.item()}')

vix_model.eval()
gold_model.eval()
with torch.no_grad():
    test_vix_outputs = vix_model(test_X)
    test_gold_outputs = gold_model(test_X)
    test_vix_loss = criterion(test_vix_outputs, test_vix_y.view(-1, 1))
    test_gold_loss = criterion(test_gold_outputs, test_gold_y.view(-1, 1))
    print(f'Test VIX Loss: {test_vix_loss.item()}')
    print(f'Test Gold Loss: {test_gold_loss.item()}')

predicted_vix_values = test_vix_outputs.detach().numpy()
predicted_gold_values = test_gold_outputs.detach().numpy()
normalized_vix = (predicted_vix_values - np.min(predicted_vix_values)) / (np.max(predicted_vix_values) - np.min(predicted_vix_values))
normalized_gold = (predicted_gold_values - np.min(predicted_gold_values)) / (np.max(predicted_gold_values) - np.min(predicted_gold_values))

plt.figure(figsize=(10, 6))
plt.plot(normalized_vix, label='Normalized Predicted VIX Close')
plt.plot(normalized_gold, label='Normalized Predicted Gold Close')
plt.legend()
plt.title('Normalized Predicted VIX and Gold Close Prices')
plt.show()
