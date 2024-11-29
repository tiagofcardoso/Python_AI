import numpy as np
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import os

data_path = os.path.join(os.path.dirname(
    __file__), './data_sets/HistoricalData_meta.csv')
df = pd.read_csv(data_path)

# removendo o símbolo de dólar
columns_to_clean = ['Close/Last', 'Volume', 'Open', 'High', 'Low']
for column in columns_to_clean:
    df[column] = df[column].astype(str).str.replace(
        '$', '', regex=False).astype(float)

data = df['Close/Last'].values.reshape(-1, 1)
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# Criar sequências


def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)


seq_length = 10
X, y = create_sequences(data, seq_length)

# Dividir dados em treino e teste
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Ajustar modelo LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=100, batch_size=32)

predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test)

mae = np.mean(np.abs(predictions - y_test))
print(f'Mean Absolute Error (MAE): {mae}')
rmae = np.sqrt(np.mean(np.abs(predictions - y_test)))
print(f'Root Mean Absolute Error (RMAE): {rmae}')


fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df.index[-len(y_test):], y=y_test.flatten(), mode='lines', name='Actual'))
fig.add_trace(go.Scatter(
    x=df.index[-len(y_test):], y=predictions.flatten(), mode='lines', name='Predicted'))
fig.update_layout(title='LSTM Forecast',
                  xaxis_title='Date', yaxis_title='Price')
fig.show()
