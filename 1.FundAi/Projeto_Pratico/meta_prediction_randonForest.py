from sklearn.ensemble import RandomForestRegressor
import numpy as np
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
df.reset_index(inplace=True)
X = df[['Date']]
y = df['Close/Last']

# Dividir dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Ajustar modelo Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Fazer previsões
predictions = model.predict(X_test)
# Calcular MAE e RMSE
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions, squared=False)

# Plotar resultados
fig = go.Figure()
fig.add_trace(go.Scatter(x=y_test.index, y=y_test,
              mode='lines', name='Actual'))
fig.add_trace(go.Scatter(x=y_test.index, y=predictions,
              mode='lines', name='Predicted'))
fig.update_layout(title='Random Forest Forecast',
                  xaxis_title='Date', yaxis_title='Price')
fig.show()

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
