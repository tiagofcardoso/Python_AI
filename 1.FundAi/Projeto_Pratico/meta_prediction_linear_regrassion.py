import os
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
# Definir variáveis X e y
X = df[['Date']].apply(lambda x: x.astype('int64') // 10**9)  # Convertendo datas para timestamp
y = df['Close/Last']

# Dividir dados em treino e teste


# Dividir dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Ajustar modelo de regressão linear
model = LinearRegression()
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
fig.update_layout(title='Linear Regression Forecast',
                  xaxis_title='Date', yaxis_title='Price')
fig.show()

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
