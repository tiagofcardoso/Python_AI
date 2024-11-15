import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# Carregar os dados
df = pd.read_csv(
    '/home/tiagocardoso/AIEngineer/1.FundAi/lab06/lab06.1/dados_receita.csv')

# Dividir os dados em características (X) e variável alvo (y)
X = df[['GastoPublicidade', 'SatisfacaoCliente',
        'NovosClientes', 'Sazonalidade']].values
y = df['ReceitaMensal'].values

# Normalizar as características para melhorar o desempenho do modelo
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Converter dados para tensores do PyTorch
X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(
    X_test, dtype=torch.float32)
y_train, y_test = torch.tensor(y_train, dtype=torch.float32), torch.tensor(
    y_test, dtype=torch.float32)

# Definir o modelo de regressão linear
# TODO: Criar o neurónio (camada linear) no modelo


class ModeloRegressaoLinear(nn.Module):
    def __init__(self, input_dim):
        super(ModeloRegressaoLinear, self).__init__()
        # TODO: Definir uma camada linear com tamanho de entrada e saída apropriados
        self.linear = nn.Linear(input_dim, 1)
        pass

    def forward(self, x):
        # TODO: Implementar a passagem de dados pela camada linear
        return self.linear(x)
        pass


# TODO: Inicializar o modelo com o número correto de características
input_dim = X_train.shape[1]
modelo = ModeloRegressaoLinear(input_dim)

# Definir a função de perda e o otimizador
criterio = nn.MSELoss()
otimizador = torch.optim.SGD(modelo.parameters(), lr=0.001)

# TODO: Treinar o modelo
n_epocas = 1000
for epoca in range(n_epocas):
    modelo.train()
    otimizador.zero_grad()
    y_pred = modelo(X_train).squeeze()
    perda = criterio(y_pred, y_train)
    perda.backward()
    otimizador.step()
    if epoca % 100 == 0:
        print(f"Época {epoca}, Perda: {perda.item()}")

# TODO: Avaliar o modelo com dados de teste e calcular métricas
# exemplo: R² e MAE com sklearn.metrics
# Avaliar o modelo
modelo.eval()
with torch.no_grad():
    y_pred_test = modelo(X_test).squeeze()
    y_pred_test_np = y_pred_test.detach().cpu().numpy()
    y_test_np = y_test.cpu().numpy()
    mae = mean_absolute_error(y_test_np, y_pred_test_np)
    print("Métricas de Desempenho do Modelo:")
    print("R²:", r2_score(y_test_np, y_pred_test_np))
    print("MAE:", mae)

# TODO: Guardar o modelo em um arquivo
nome_arquivo = 'modelo_treinado.pth'
output_dir = '/home/tiagocardoso/AIEngineer/1.FundAi/lab06/lab06.1/'
torch.save(modelo.state_dict(), output_dir + nome_arquivo)
print(f"Modelo salvo em {output_dir}{nome_arquivo}")
