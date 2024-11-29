import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, Dropout
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
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

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Preparar dados
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

# Ajustar modelo Transformer

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


embed_dim = 32
num_heads = 2
ff_dim = 32

inputs = Input(shape=(seq_length, 1))
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(inputs, training=True)
x = Dense(1)(x)
model = Model(inputs=inputs, outputs=x)

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=20, batch_size=32)

# Fazer previsões
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
y_test = scaler.inverse_transform(y_test)

# Plotar resultados
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df.index[-len(y_test):], y=y_test.flatten(), mode='lines', name='Actual'))
fig.add_trace(go.Scatter(
    x=df.index[-len(y_test):], y=predictions.flatten(), mode='lines', name='Predicted'))
fig.update_layout(title='Transformer Forecast',
                  xaxis_title='Date', yaxis_title='Price')
fig.show()
