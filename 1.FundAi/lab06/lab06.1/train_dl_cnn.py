import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# Carregar os dados
df = pd.read_csv('/home/tiagocardoso/AIEngineer/1.FundAi/lab06/lab06.1/dados_receita.csv')

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
# Criar o neurónio (camada linear) no modelo - 1 neurónio


class ModeloRegressaoLinear(nn.Module):
    def __init__(self, input_dim):
        super(ModeloRegressaoLinear, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # 1 neuronio

    def forward(self, x):
        return self.linear(x)

# Criar rede neural com diferentes configuracões


class ModeloRegressaoFlexivel(nn.Module):
    def __init__(self, input_dim, hidden_layers):
        super(ModeloRegressaoFlexivel, self).__init__()
        layers = []
        in_features = input_dim
        for neurons in layers:
            layers.append(nn.Linear(in_features, neurons))
            layers.append(nn.ReLU())
            in_features = neurons
        layers.append(nn.Linear(in_features, 1))  # Saida
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def treinar_e_avaliar_modelo(hidden_layers, input_dim, X_train, y_train, X_test, y_test):
    # Inicializar o modelo com o número correto de características
    modelo = ModeloRegressaoFlexivel(input_dim, hidden_layers)

    # Definir a função de perda e o otimizador
    criterio = nn.MSELoss()
    # taxa de aprendizagem muito baixa
    otimizador = torch.optim.SGD(modelo.parameters(), lr=0.001)

    # Treinar o modelo
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

    return modelo, mae


melhor_mae = float('inf')
melhor_modelo = None
estruras_testadas = [[16], [32], [64], [128]]
input_dim = X_train.shape[1]

for estrutura in estruras_testadas:
    modelo, mae = treinar_e_avaliar_modelo(
        estrutura, input_dim, X_train, y_train, X_test, y_test)
    if mae < melhor_mae:
        melhor_mae = mae
        melhor_modelo = modelo
        melhor_estrutura = estrutura

        # Salvar o modelo atual como o melhor até agora com nome descritivo
        nome_arquivo = f"modelo_melhor_{melhor_estrutura}_MAE_{mae:.2f}.pth"
        output_dir = '/home/tiagocardoso/AIEngineer/1.FundAi/lab06/lab06.1/web/'
        torch.save(melhor_modelo.state_dict(), output_dir + nome_arquivo)
        print(f"\nNovo melhor modelo encontrado e salvo como '{
              nome_arquivo}' com estrutura {melhor_estrutura} - MAE: {melhor_mae:.2f}")

print(f"\nMelhor estrutura final: {melhor_estrutura} - MAE: {melhor_mae:.2f}")
