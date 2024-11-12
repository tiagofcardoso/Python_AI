# pip install matplotlib

import torch  # Framework principal de deep learning
import torch.nn as nn  # Módulo de redes neurais
import matplotlib.pyplot as plt  # Biblioteca para visualização de dados

# ============ PREPARAÇÃO DOS DADOS ============
# Criação de dados sintéticos para treino
# randn gera números aleatórios com distribuição normal (média 0, desvio padrão 1)
X_train = torch.randn(100, 10)  # 100 amostras, cada uma com 10 features
# randint gera números inteiros aleatórios (0 ou 1) para classificação binária
# .float() converte para float32 pois o modelo trabalha com floats
y_train = torch.randint(0, 2, (100, 1)).float()  # 100 labels (0 ou 1)

# Dados de validação (mesma estrutura que os dados de treino, mas menor quantidade)
X_val = torch.randn(30, 10)     # 30 amostras para validação
y_val = torch.randint(0, 2, (30, 1)).float()     # Labels de validação

# ============ ARQUITETURA DO MODELO ============
# Criação de uma rede neural sequencial (camadas em sequência)
model = nn.Sequential(
    nn.Linear(10, 5),    # Primeira camada: 10 entradas -> 5 neurónios
    nn.ReLU(),          # Função de ativação ReLU após primeira camada
    nn.Linear(5, 1),    # Segunda camada: 5 neurónios -> 1 saída
    nn.Sigmoid()        # Sigmoid na saída (mapeia para valores entre 0 e 1)
)

# ============ DEFINIÇÃO DE LOSS E OTIMIZADOR ============
# Binary Cross Entropy Loss - apropriada para classificação binária
criterion = nn.BCELoss()
# Otimizador Adam com learning rate de 0.01
# Adam é geralmente mais eficiente que SGD simples
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ============ VARIÁVEIS PARA MONITORIZAÇÃO ============
# Listas para guardar histórico de loss durante o treino
train_losses = []  # Armazena losses de treino
val_losses = []    # Armazena losses de validação
best_val_loss = float('inf')  # Melhor loss de validação (começa como infinito)
epochs_without_improvement = 0  # Contador para early stopping
best_model_state = None  # Guarda os melhores pesos do modelo

# ============ PARÂMETROS DE TREINO ============
max_epochs = 1000  # Número máximo de epochs permitido
patience = 20      # Número de epochs sem melhoria antes de parar

print("A Iniciar treino...")

# ============ LOOP DE TREINO ============
for epoch in range(max_epochs):
    # === FASE DE TREINO ===
    model.train()  # Coloca o modelo em modo de treino (ativa gradientes)
    # Forward pass
    y_pred = model(X_train)  # Gera previsões
    train_loss = criterion(y_pred, y_train)  # Calcula erro de treino

    # Backward pass
    optimizer.zero_grad()  # Zera gradientes acumulados
    train_loss.backward()  # Calcula gradientes
    optimizer.step()      # Atualiza pesos

    # === FASE DE VALIDAÇÃO ===
    model.eval()  # Coloca o modelo em modo de avaliação (desativa gradientes)
    with torch.no_grad():  # Contexto onde gradientes não são calculados
        y_val_pred = model(X_val)  # Previsões no conjunto de validação
        val_loss = criterion(y_val_pred, y_val)  # Calcula erro de validação

    # === ARMAZENAMENTO DE MÉTRICAS ===
    # Guarda os valores de loss para plotagem posterior
    # .item() converte tensor para número
    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())

    # === EARLY STOPPING CHECK ===
    if val_loss < best_val_loss:  # Se encontrou melhor resultado
        best_val_loss = val_loss  # Atualiza melhor loss
        epochs_without_improvement = 0  # Reinicia contador
        best_model_state = model.state_dict()  # Guarda melhores pesos
    else:
        epochs_without_improvement += 1  # Incrementa contador

    # === MONITORIZAÇÃO DE PROGRESSO ===
    # Imprime status a cada 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{max_epochs}]')
        print(f'Train Loss: {train_loss.item():.4f}')
        print(f'Val Loss: {val_loss.item():.4f}')

    # === VERIFICAÇÃO DE EARLY STOPPING ===
    # Para o treino se não houver melhoria por 'patience' epochs
    if epochs_without_improvement >= patience:
        print(f'\nEarly stopping na epoch {epoch+1}')
        break

# ============ VISUALIZAÇÃO DOS RESULTADOS ============
# Configuração do gráfico
plt.figure(figsize=(10, 6))  # Tamanho da figura: 10x6 polegadas
# Plota as curvas de loss
plt.plot(train_losses, label='Train Loss')  # Loss de treino em azul
plt.plot(val_losses, label='Validation Loss')  # Loss de validação em laranja
# Configurações do gráfico
plt.xlabel('Epochs')  # Rótulo do eixo x
plt.ylabel('Loss')    # Rótulo do eixo y
plt.title('Train vs Validation Loss')  # Título do gráfico
plt.legend()          # Mostra legenda
plt.grid(True)        # Adiciona grade
plt.show()            # Mostra o gráfico

# Imprime a melhor loss de validação alcançada
print(f"\nMelhor validation loss: {best_val_loss:.4f}")
