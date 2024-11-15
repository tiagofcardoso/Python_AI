from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import joblib
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Carregar os dados
df = pd.read_csv(
    '/home/tiagocardoso/AIEngineer/1.FundAi/lab06/dados_receita.csv')

# Dividir os dados
X = df[['GastoPublicidade', 'SatisfacaoCliente', 'NovosClientes', 'Sazonalidade']]
y = df['ReceitaMensal']
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42)

# Dicionário para armazenar modelos e métricas
model_performance = {}

# Seção para Regressão Linear Simples
# Treinar o modelo
print("Treinando Regressão Linear Simples...")
lr_limple = LinearRegression()
lr_limple.fit(X_train[['GastoPublicidade']], y_train)
y_val_pred_simple = lr_limple.predict(X_val[['GastoPublicidade']])
r2_simple = r2_score(y_val, y_val_pred_simple)
mae_simple = mean_absolute_error(y_val, y_val_pred_simple)
model_performance['Regressão Linear Simples'] = (
    lr_limple, r2_simple, mae_simple)

print("R² de Validação:", r2_simple)
print("MAE de Validação:", mae_simple)
print()

# Seção para Regressão Linear Múltipla
print("Treinando Regressão Linear Multipla...")
lr_multi = LinearRegression()
lr_multi.fit(X_train, y_train)
y_val_pred_multi = lr_multi.predict(X_val)
r2_multi = r2_score(y_val, y_val_pred_multi)
mae_multi = mean_absolute_error(y_val, y_val_pred_multi)
model_performance['Regressão Linear Multipla'] = (
    lr_multi, r2_multi, mae_multi)

print("R² de Validação:", r2_multi)
print("MAE de Validação:", mae_multi)
print()

# Seção para Regressão Polinomial
# Treinar o modelo
print("Treinando Regressão Polinomial (grau 2)...")
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly_features.fit_transform(
    X_train[['GastoPublicidade', 'SatisfacaoCliente']])
X_val_poly = poly_features.transform(
    X_val[['GastoPublicidade', 'SatisfacaoCliente']])
lr_poly = LinearRegression()
lr_poly.fit(X_train_poly, y_train)
y_val_pred_poly = lr_poly.predict(X_val_poly)
r2_poly = r2_score(y_val, y_val_pred_poly)
mae_poly = mean_absolute_error(y_val, y_val_pred_poly)
model_performance['Regressão Polinomial'] = (lr_poly, r2_poly, mae_poly)

print("R² de Validação:", r2_poly)
print("MAE de Validação:", mae_poly)
print()

# Seção para Regressão Lasso
print("Treinando Regressão Lasso...")
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
y_val_pred_lasso = lasso.predict(X_val)
r2_lasso = r2_score(y_val, y_val_pred_lasso)
mae_lasso = mean_absolute_error(y_val, y_val_pred_lasso)
model_performance['Regressão Lasso'] = (lasso, r2_lasso, mae_lasso)

print("R² de Validação:", r2_lasso)
print("MAE de Validação:", mae_lasso)
print()


# Seção para Regressão Ridge
print("Treinando Regressão Ridge...")
ridge = Ridge(alpha=0.1)
ridge.fit(X_train, y_train)
y_val_pred_ridge = ridge.predict(X_val)
r2_ridge = r2_score(y_val, y_val_pred_ridge)
mae_ridge = mean_absolute_error(y_val, y_val_pred_ridge)
model_performance['Regressão Ridge'] = (ridge, r2_ridge, mae_ridge)

print("R² de Validação:", r2_ridge)
print("MAE de Validação:", mae_ridge)
print()

# TODO: Selecionar o melhor modelo com base no R² de validação
print("Selecionando o melhor modelo...")
#print(model_performance)
melhor_nome, (melhor_modelo, melhor_r2, melhor_mae) = max(model_performance.items(), key=lambda x: x[1][1])
print("Melhor modelo:", melhor_nome)

# TODO: Exportar o melhor modelo com o MAE no nome do arquivo
joblib.dump(lr_limple, 'lr_limple.pkl')
nome_arquivo = f"{melhor_nome}_MAE_{melhor_mae:.2f}.pkl"
joblib.dump(melhor_modelo, nome_arquivo)
