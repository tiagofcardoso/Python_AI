from sklearn.linear_model import Lasso
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# Carregar o dataset
data = fetch_california_housing()
x = data.data
y = data.target

# Dividir o dataset em treino e teste
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42)


# Treinar o modelo
model = LinearRegression()
model.fit(x_train, y_train)

# Fazer previsões
y_pred = model.predict(x_test)

# Calcular métricas de erro
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)


print(f"Erro Médio Absoluto (MAE): {mae:.2f}")
print(f"Erro Quadrático Médio (MSE): {mse:.2f}")
print(f"Raiz do Erro Quadrático Médio (RMSE): {rmse:.2f}")
print(f"Coeficiente de Determinação (R²): {r2:.2f}")


# Testar diferentes valores de alpha para regularização
alphas = [0.001, 0.01, 0.1, 1, 10]
results = []
for alpha in alphas:
    model = Lasso(alpha=alpha)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)    
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    results.append([alpha, mae, mse, rmse, r2])

# Converter resultados em um DataFrame
results_df = pd.DataFrame(
    results, columns=['Alpha', 'MAE', 'MSE', 'RMSE', 'R²'])
print(results_df)
    
# Visualizar as métricas para diferentes valores de alpha
plt.figure(figsize=(12, 8))
plt.plot(results_df['Alpha'], results_df['MAE'], label='MAE', marker='o')
plt.plot(results_df['Alpha'], results_df['MSE'], label='MSE', marker='o')
plt.plot(results_df['Alpha'], results_df['RMSE'], label='RMSE', marker='o')
plt.plot(results_df['Alpha'], results_df['R²'], label='R²', marker='o')
plt.xscale('log')
plt.xlabel('Alpha (Regularização)')
plt.ylabel('Valor da Métrica')
plt.title('Variação das Métricas com Diferentes Valores de Alpha')
plt.legend()
plt.show()