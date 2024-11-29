from arch import arch_model
import plotly.graph_objs as go
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import os
from sklearn.metrics import mean_absolute_error
import numpy as np

# Carregar dados
data_path = os.path.join(os.path.dirname(
    __file__), './data_sets/HistoricalData_meta.csv')
data = pd.read_csv(data_path)

# removendo o símbolo de dólar
columns_to_clean = ['Close/Last', 'Volume', 'Open', 'High', 'Low']
for column in columns_to_clean:
    data[column] = data[column].astype(str).str.replace(
        '$', '', regex=False).astype(float)

data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Close/Last é a coluna que contém os preços de fechamento das ações
df = data[['Close/Last']].reset_index()
df.columns = ['ds', 'y']


# Ajustar modelo GARCH
model = arch_model(df['y'], vol='Garch', p=1, q=1)
model_fit = model.fit()

# Fazer previsões
forecast = model_fit.forecast(horizon=10)
forecast_variance = forecast.variance.values[-1, :]
forecast_dates = pd.date_range(start=df.index[-1], periods=11, inclusive='right')

# Plotar resultados
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df.index, y=df['y'], mode='lines', name='Actual'))
fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_variance,
              mode='lines', name='Forecast Variance'))
fig.update_layout(title='GARCH Forecast',
                  xaxis_title='Date', yaxis_title='Variance')
fig.show()

# Calcular MAE
mae = mean_absolute_error(df['y'], model_fit.conditional_volatility)
print(f'Mean Absolute Error (MAE): {mae}')

# Calcular RMAE
rmae = np.sqrt(mae)
print(f'Root Mean Absolute Error (RMAE): {rmae}')


# Criar figura
fig = go.Figure()

# Adicionar trace para os dados históricos
fig.add_trace(go.Scatter(x=df['ds'], y=df['y'],
              mode='lines', name='Histórico'))

# Adicionar trace para as previsões
future_dates = pd.date_range(
    start=df['ds'].iloc[-1], periods=11, inclusive='both')
forecast_mean = forecast.mean.values[-1, :]
fig.add_trace(go.Scatter(x=future_dates, y=forecast_mean,
              mode='lines', name='Previsão'))

# Atualizar layout
fig.update_layout(title='Previsão de Preços das Ações',
                  xaxis_title='Data',
                  yaxis_title='Preço de Fechamento',
                  template='plotly_dark')

fig.show()
