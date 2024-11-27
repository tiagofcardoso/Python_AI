import re
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from prophet.plot import plot_plotly, plot_components_plotly

data = pd.read_csv(
    # replace you path here
    '/home/tiagocardoso/AIEngineer/1.FundAi/Projeto_Pratico/data_sets/financial_data.csv')

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
