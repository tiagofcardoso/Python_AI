import re
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from prophet.plot import plot_plotly, plot_components_plotly
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import os


data_path = os.path.join(os.path.dirname(__file__), './data_sets/financial_data.csv')
data = pd.read_csv(data_path)

data.dropna(inplace=True)  # limpando dados nulos
# removendo colunas desnecessárias
data = data.drop(['weather_temperature', 'social_media_mentions'], axis=1)

# convertendo a coluna de timestamp para o formato datetime
data['timestamp'] = pd.to_datetime(data['timestamp'])

# definindo a coluna de data como índice
data.set_index('timestamp', inplace=True)

# bid_ask_spread é a coluna que contém os preços de fechamento das ações
df = data[['bid_ask_spread']].reset_index()
df.columns = ['ds', 'y']

# Prophet com hiperparâmetros personalizados
# model = Prophet(
#     seasonality_mode='multiplicative',
#     yearly_seasonality=True,
#     weekly_seasonality=True,
#     daily_seasonality=False,
#     changepoint_prior_scale=0.05
# )

model = Prophet()
model.fit(df)

future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

# Calculate a simple moving average (SMA) for trend identification
forecast['SMA'] = forecast['yhat'].rolling(window=10).mean()
print(f'SMA:{forecast}')

# Define a simple trading strategy based on the SMA and predicted values
forecast['Signal'] = 0
forecast.loc[forecast['yhat'] > forecast['SMA'], 'Signal'] = 1  # Buy signal
forecast.loc[forecast['yhat'] < forecast['SMA'], 'Signal'] = -1  # Sell signal

# Save the forecast and trading signals to a CSV file
# forecast.to_csv('/home/tiagocardoso/AIEngineer/1.FundAi/Projeto_Pratico/data_sets/forecast.csv')

# Calculate MAE between the actual and predicted values
actual = df['y']
predicted = forecast.loc[forecast['ds'].isin(df['ds']), 'yhat']
mae = mean_absolute_error(actual, predicted)
rmse = mean_squared_error(actual, predicted, squared=False) ** 0.5

# Print the MAE
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Root Mean Squared Error (RMSE): {rmse}')

# Plot the forecast and components using Plotly
fig = plot_plotly(model, forecast, xlabel='Date', ylabel='Price')
fig.show()

# Plot the forecast components using Plotly
fig_components = plot_components_plotly(model, forecast)
fig_components.show()

#############################################################################################################################################################################
# Prepare the data for RNN
data = data[['bid_ask_spread']].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create the training and test datasets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Create the dataset with look_back
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 10
X_train, y_train = create_dataset(train_data, look_back)
X_test, y_test = create_dataset(test_data, look_back)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(2)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(2)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1])
        return predictions

model = LSTMModel()
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 20
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = loss_function(y_pred, y_train)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# Make predictions
model.eval()
train_predict = model(X_train).detach().numpy()
test_predict = model(X_test).detach().numpy()

# Invert predictions
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform(y_train.detach().numpy())
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform(y_test.detach().numpy())

# Calculate root mean squared error
train_score = np.sqrt(mean_squared_error(y_train, train_predict))
print(f'Train Score: {train_score} RMSE')
test_score = np.sqrt(mean_squared_error(y_test, test_predict))
print(f'Test Score: {test_score} RMSE')

# Shift train predictions for plotting
train_predict_plot = np.empty_like(scaled_data)
train_predict_plot[:, :] = np.nan
train_predict_plot[look_back:len(train_predict) + look_back, :] = train_predict

# Shift test predictions for plotting
test_predict_plot = np.empty_like(scaled_data)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict) + (look_back * 2) + 1:len(scaled_data) - 1, :] = test_predict


import plotly.graph_objs as go

# Create traces for the actual, train, and test predictions
trace_actual = go.Scatter(
    x=df['ds'],
    y=scaler.inverse_transform(scaled_data).flatten(),
    mode='lines',
    name='Actual'
)

trace_train = go.Scatter(
    x=df['ds'][look_back:len(train_predict) + look_back],
    y=train_predict.flatten(),
    mode='lines',
    name='Train Predict'
)

trace_test = go.Scatter(
    x=df['ds'][len(train_predict) + (look_back * 2) + 1:len(scaled_data) - 1],
    y=test_predict.flatten(),
    mode='lines',
    name='Test Predict'
)

# Create the figure and add the traces
fig = go.Figure()
fig.add_trace(trace_actual)
fig.add_trace(trace_train)
fig.add_trace(trace_test)

# Update layout
fig.update_layout(
    title='Stock Price Prediction',
    xaxis_title='Date',
    yaxis_title='Price',
    showlegend=True
)

# Show the plot
fig.show()
# Plot the forecast components using Plotly
fig_components = plot_components_plotly(model, forecast)
fig_components.show()


