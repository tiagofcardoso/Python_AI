import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Carregar os dados
df = pd.read_csv('/home/tiagocardoso/AIEngineer/1.FundAi/lab06/lab06.3/dados_demanda_energia.csv',
                 parse_dates=['Data'], index_col='Data')

# Visualizar as primeiras linhas e a série temporal
print(df.head())
df.plot(figsize=(14, 6), title='Demanda de Energia')
plt.show()
plt.show()

# Ex 1: Suavização Exponencial
modelo_exp = ExponentialSmoothing(df['Demanda'], trend='add', seasonal='add', seasonal_periods=12).fit()
previsao_exp = modelo_exp.forecast(12)
df['Demanda'].plot(label='Observado')
previsao_exp.plot(label='Previsão (Suavização Exponencial)', legend=True)
plt.show()

# Ex 2: Modelação com ARIMA
modelo_arima = ARIMA(df['Demanda'], order=(5, 1, 2)).fit()
previsao_arima = modelo_arima.forecast(steps=12)
df['Demanda'].plot(label='Observado')
previsao_arima.plot(label='Previsão (ARIMA)', legend=True)
plt.show()

# Ex 3: Modelação com SARIMA
modelo_sarima = SARIMAX(df['Demanda'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit()
previsao_sarima = modelo_sarima.forecast(steps=12)
df['Demanda'].plot(label='Observado')
previsao_sarima.plot(label='Previsão (SARIMA)', legend=True)
plt.show()

# Ex 4: Rede Neural LSTM
scaler = MinMaxScaler()
demanda_normalizada = scaler.fit_transform(df['Demanda'].values.reshape(-1, 1))

def criar_sequencias(data, seq_length):
    X, Y = [], []
    for i in range(len(data)-seq_length):
        X.append(data[i:i+seq_length])
        Y.append(data[i+seq_length])
    return np.array(X), np.array(Y)

seq_length = 12
X, Y = criar_sequencias(demanda_normalizada, seq_length)
X_train, X_test = X[:int(len(X)*0.8)], X[int(len(X)*0.8):]
Y_train, Y_test = Y[:int(len(Y)*0.8)], Y[int(len(Y)*0.8):]

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super(LSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size), torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]
    
# Ex 5: Modelo Prophet
df_prophet = df.reset_index().rename(columns={'Data': 'ds', 'Demanda': 'y'})
modelo_prophet = Prophet()
modelo_prophet.add_regressor('Temperatura')
modelo_prophet.add_regressor('Humidade')

# Ajustar o modelo Prophet com os dados e regressores
modelo_prophet.fit(df_prophet)

# Criar o dataframe de futuro e incluir os regressores
futuro = modelo_prophet.make_future_dataframe(periods=12, freq='MS')
futuro['Temperatura'] = df_prophet['Temperatura'].fillna(df_prophet['Temperatura'].mean())
futuro['Humidade'] = df_prophet['Humidade'].fillna(df_prophet['Humidade'].mean())

# Verificar e preencher valores NaN nas colunas de regressores
futuro['Temperatura'].fillna(df_prophet['Temperatura'].mean(), inplace=True)
futuro['Humidade'].fillna(df_prophet['Humidade'].mean(), inplace=True)

# Gerar a previsão com o modelo ajustado
previsao_prophet = modelo_prophet.predict(futuro)
modelo_prophet.plot(previsao_prophet)
plt.title('Previsão com Prophet')
plt.show()

# Ex 6: Comparação de Modelos
mae_exp = mean_absolute_error(df['Demanda'][-12:], previsao_exp)
rmse_exp = mean_squared_error(df['Demanda'][-12:], previsao_exp, squared=False)
mae_arima = mean_absolute_error(df['Demanda'][-12:], previsao_arima)
rmse_arima = mean_squared_error(df['Demanda'][-12:], previsao_arima, squared=False)
mae_sarima = mean_absolute_error(df['Demanda'][-12:], previsao_sarima)
rmse_sarima = mean_squared_error(df['Demanda'][-12:], previsao_sarima, squared=False)
mae_prophet = mean_absolute_error(df['Demanda'][-12:], previsao_prophet['yhat'][-12:])
rmse_prophet = mean_squared_error(df['Demanda'][-12:], previsao_prophet['yhat'][-12:], squared=False)

print("Suavização Exponencial - MAE:", mae_exp, "RMSE:", rmse_exp)
print("ARIMA - MAE:", mae_arima, "RMSE:", rmse_arima)
print("SARIMA - MAE:", mae_sarima, "RMSE:", rmse_sarima)
print("Prophet - MAE:", mae_prophet, "RMSE:", rmse_prophet)

# Ex 7: Teste com Dia de Mau Tempo
# Definir um dia de evento climático extremo (mau tempo)
mau_tempo = {
    'ds': pd.to_datetime('2023-12-25'),
    'Temperatura': -5,
    'Humidade': 90,
}
# Adicionar o evento ao futuro dataframe para previsão
futuro_com_mau_tempo = futuro.copy()
futuro_com_mau_tempo.loc[futuro_com_mau_tempo['ds'] == '2023-12-25', ['Temperatura', 'Humidade']] = [-5, 90]

# Prever com Prophet incluindo o evento climático extremo
previsao_prophet_mau_tempo = modelo_prophet.predict(futuro_com_mau_tempo)

# Visualizar o impacto do evento climático na previsão
modelo_prophet.plot(previsao_prophet_mau_tempo)
plt.title('Impacto do Mau Tempo na Previsão - Prophet')
plt.show()

# Comparação das previsões com e sem mau tempo
# Comparação das previsões com e sem mau tempo
print("\nComparação de Previsões para 2023-12-25 com e sem mau tempo:")
print("Previsão normal (kWh):", previsao_prophet[previsao_prophet['ds'] == '2023-12-25'][['ds', 'yhat']].rename(columns={'yhat': 'Demanda (kWh)'}))
print("Previsão com mau tempo (kWh):", previsao_prophet_mau_tempo[previsao_prophet_mau_tempo['ds'] == '2023-12-25'][['ds', 'yhat']].rename(columns={'yhat': 'Demanda (kWh)'}))
