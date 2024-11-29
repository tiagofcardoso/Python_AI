from prophet.diagnostics import cross_validation, performance_metrics
import re
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from prophet.plot import plot_plotly, plot_components_plotly
import os

data_path = os.path.join(os.path.dirname(__file__), './data_sets/HistoricalData_AAPL.csv')
data = pd.read_csv(data_path)
print(data.columns)

# removendo o símbolo de dólar
data['Close/Last'] = data['Close/Last'].str.replace('$', '', regex=False).astype(float)

# convertendo a coluna de data para o formato datetime
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)  # definindo a coluna de data como índice

# Close/Last é a coluna que contém os preços de fechamento das ações
df = data[['Close/Last']].reset_index()
df.columns = ['ds', 'y']

model = Prophet(
    seasonality_mode='multiplicative',
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.05
)
model.fit(df)

future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

# Calculate a simple moving average (SMA) for trend identification
forecast['SMA'] = forecast['yhat'].rolling(window=10).mean()

# Define a simple trading strategy based on the SMA and predicted values
forecast['Signal'] = 0
forecast.loc[forecast['yhat'] > forecast['SMA'], 'Signal'] = 1  # Buy signal
forecast.loc[forecast['yhat'] < forecast['SMA'], 'Signal'] = -1  # Sell signal

# Save the forecast and trading signals to a CSV file
forecast.to_csv(
    '/home/tiagocardoso/AIEngineer/1.FundAi/Projeto_Pratico/data_sets/forecast_apple.csv')

# Calculate MAE between the actual and predicted values
actual = df['y']
predicted = forecast.loc[forecast['ds'].isin(df['ds']), 'yhat']
mae = mean_absolute_error(actual, predicted)
rmse = mean_squared_error(actual, predicted, squared=False) ** 0.5

# Print the MAE
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Root Mean Squared Error (RMSE): {rmse}')


df_cv = cross_validation(model, initial='730 days',
                         period='180 days', horizon='365 days')
df_p = performance_metrics(df_cv)
print(df_p[['mae', 'rmse']])

# Plot the forecast and components using Plotly
fig = plot_plotly(model, forecast, xlabel='Date', ylabel='Price')
fig.update_layout(title='APPLE')
fig.show()

# Plot the forecast components using Plotly
fig_components = plot_components_plotly(model, forecast)
fig_components.update_layout(title='APPLE split')
fig_components.show()
