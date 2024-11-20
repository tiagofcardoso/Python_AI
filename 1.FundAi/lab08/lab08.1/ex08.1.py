import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from collections import deque
import random
from sklearn.metrics import mean_squared_error, accuracy_score

print("Lab 08.1 - Algoritmos de Reforço\n")

# Ex 1: Limite de Confiança Superior (UCB)
print("\nExecutar Ex 1: UCB")

# Definições
n_channels = 5
n_rounds = 1000
rewards = np.zeros(n_channels)
counts = np.zeros(n_channels)

# Função para calcular UCB
def ucb(counts, rewards, round_number):
    return rewards + np.sqrt(2 * np.log(round_number + 1) / (counts + 1e-5))

# Ciclo de interação
total_rewards = []
for round_number in range(n_rounds):
    if round_number < n_channels:
        channel = round_number
    else:
        channel = np.argmax(ucb(counts, rewards, round_number))
    reward = np.random.binomial(1, 0.5)  # Recompensa binária (conversão ou não)
    counts[channel] += 1
    rewards[channel] = (rewards[channel] * (counts[channel] - 1) + reward) / counts[channel]
    total_rewards.append(np.sum(rewards))

# Visualizar resultados
plt.figure(figsize=(10, 5))
plt.plot(total_rewards)
plt.xlabel('Rodadas')
plt.ylabel('Recompensa Total')
plt.title('Evolução das Recompensas no Algoritmo UCB')
plt.show()

# Métricas
print("\nMétricas UCB:")
recompensa_cumulativa = np.sum(rewards)
print(f'Recompensa Cumulativa: {recompensa_cumulativa}')
taxa_convergencia = np.mean(total_rewards[-100:])
print(f'Taxa de Convergência: {taxa_convergencia}')
tempo_exploracao = np.sum(counts < (n_rounds / n_channels))
tempo_exploracao_ratio = tempo_exploracao / n_rounds
print(f'Tempo de Exploração: {tempo_exploracao} ({tempo_exploracao_ratio:.2%} do total)')

# Ex 2: Amostragem de Thompson
print("\nExecutar Ex 2: Thompson Sampling")

# Definições
successes = np.zeros(n_channels)
failures = np.zeros(n_channels)

# Ciclo de interação
total_rewards = []
for round_number in range(n_rounds):
    samples = [np.random.beta(successes[i] + 1, failures[i] + 1) for i in range(n_channels)]
    channel = np.argmax(samples)
    reward = np.random.binomial(1, 0.5)  # Recompensa binária (venda ou não)
    if reward == 1:
        successes[channel] += 1
    else:
        failures[channel] += 1
    total_rewards.append(np.sum(successes))

# Visualizar resultados
plt.figure(figsize=(10, 5))
plt.plot(total_rewards)
plt.xlabel('Rodadas')
plt.ylabel('Recompensa Total')
plt.title('Evolução das Recompensas na Amostragem de Thompson')
plt.show()

# Métricas
print("\nMétricas Thompson Sampling:")
recompensa_cumulativa = np.sum(successes)
print(f'Recompensa Cumulativa: {recompensa_cumulativa}')
taxa_convergencia = np.mean(total_rewards[-100:])
print(f'Taxa de Convergência: {taxa_convergencia}')
tempo_exploracao = np.sum(failures)
tempo_exploracao_ratio = tempo_exploracao / n_rounds
print(f'Tempo de Exploração: {tempo_exploracao} ({tempo_exploracao_ratio:.2%} do total)')

# Ex 3: Q-Learning
print("\nExecutar Ex 3: Q-Learning")

# Definições
grid_size = 5
q_table = np.zeros((grid_size, grid_size, 4))  # Q-table para 4 ações
alpha = 0.1  # Taxa de aprendizagem
gamma = 0.9  # Fator de desconto
epsilon = 0.1  # Probabilidade de exploração
n_episodes = 1000

# Definir ambiente (representação simplificada de um mapa)
movements = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Cima, Baixo, Esquerda, Direita

def select_action(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(4))
    return np.argmax(q_table[state])

def update_state(state, action):
    next_state = (state[0] + movements[action][0], state[1] + movements[action][1])
    next_state = (max(0, min(next_state[0], grid_size - 1)), max(0, min(next_state[1], grid_size - 1)))
    return next_state

# Treinar o agente
episode_rewards = []
for episode in range(n_episodes):
    state = (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))
    total_reward = 0
    done = False

    while not done:
        action = select_action(state, epsilon)
        next_state = update_state(state, action)
        reward = 1 if next_state == (grid_size - 1, grid_size - 1) else -0.1

        q_table[state][action] = (1 - alpha) * q_table[state][action] + alpha * (reward + gamma * np.max(q_table[next_state]))

        state = next_state
        total_reward += reward
        done = (state == (grid_size - 1, grid_size - 1))
    
    episode_rewards.append(total_reward)

# Métricas
print("\nMétricas Q-Learning:")
recompensa_cumulativa = np.sum(q_table)
print(f'Recompensa Cumulativa: {recompensa_cumulativa}')
taxa_convergencia = np.mean(q_table)
print(f'Taxa de Convergência: {taxa_convergencia}')
tempo_exploracao = np.sum([1 for i in range(grid_size) for j in range(grid_size) for k in range(4) if q_table[i, j, k] == 0])
tempo_exploracao_ratio = tempo_exploracao / (grid_size * grid_size * 4)
print(f'Tempo de Exploração: {tempo_exploracao} ({tempo_exploracao_ratio:.2%} do total)')


# Ex 4: Deep Q-Learning
print("\nExecutar Ex 4: Deep Q-Learning")

# Definir o ambiente
env = gym.make('CartPole-v1')

# Definir a rede neural em PyTorch
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Inicializar rede neural, otimizador e função de perda
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
policy_net = DQN(input_dim, output_dim)
target_net = DQN(input_dim, output_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Definir hiperparâmetros
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 32
memory = deque(maxlen=2000)

def choose_action(state):
    if np.random.rand() <= epsilon:
        return env.action_space.sample()
    with torch.no_grad():
        state = torch.FloatTensor(state).unsqueeze(0)
        return torch.argmax(policy_net(state)).item()

def replay():
    if len(memory) < batch_size:
        return
    minibatch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*minibatch)

    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions).unsqueeze(1)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)

    current_q_values = policy_net(states).gather(1, actions).squeeze()
    next_q_values = target_net(next_states).max(1)[0]
    expected_q_values = rewards + (gamma * next_q_values * (1 - dones))

    loss = criterion(current_q_values, expected_q_values.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Treinar o agente
n_episodes = 200
episode_rewards = []
for e in range(n_episodes):
    state, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0

    while not (done or truncated):
        action = choose_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward
        replay()

    episode_rewards.append(total_reward)
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    if e % 10 == 0:
        target_net.load_state_dict(policy_net.state_dict())
        print(f"Episódio: {e}/{n_episodes}, Recompensa: {total_reward}, Epsilon: {epsilon:.2}")

env.close()

# Métricas
print("\nMétricas Deep Q-Learning:")
recompensa_cumulativa = np.sum(episode_rewards)
print(f'Recompensa Cumulativa: {recompensa_cumulativa}')
taxa_convergencia = np.mean(episode_rewards[-10:])
print(f'Taxa de Convergência: {taxa_convergencia}')
print(f'Tempo de Exploração vs. Explotação: {epsilon_decay:.2%}')

# Ex 5: PPO
print("\nExecutar Ex 5: PPO")

# Inicializar o ambiente
env = gym.make('CartPole-v1')

# Definir a rede neural do PPO
class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCritic, self).__init__()
        # Rede do Ator (política)
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, output_dim),
            nn.Softmax(dim=-1)
        )
        # Rede do Crítico (valor)
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)

# Inicializar modelo e otimizador
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
model = ActorCritic(input_dim, output_dim)
optimizer = optim.Adam(model.parameters(), lr=0.002)

# Hiperparâmetros
n_episodes = 100
clip_epsilon = 0.2
c1 = 1.0  # Coeficiente da perda do valor
c2 = 0.01  # Coeficiente da entropia

def compute_returns(rewards, gamma=0.99):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

episode_rewards = []
for episode in range(n_episodes):
    state, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0
    
    states, actions, rewards, log_probs = [], [], [], []
    
    while not (done or truncated):
        state_tensor = torch.FloatTensor(state)
        
        # Obter ação e valor
        action_probs, state_value = model(state_tensor)
        action = torch.multinomial(action_probs, 1)
        log_prob = torch.log(action_probs[action.item()]).item()
        
        # Executar ação no ambiente
        next_state, reward, done, truncated, _ = env.step(action.item())
        
        # Armazenar transição
        states.append(state_tensor)
        actions.append(action.item())
        rewards.append(reward)
        log_probs.append(log_prob)
        
        state = next_state
        total_reward += reward

    episode_rewards.append(total_reward)
    
    # Calcular retornos
    returns = torch.tensor(compute_returns(rewards))
    states_tensor = torch.stack(states)
    actions_tensor = torch.tensor(actions)
    old_log_probs = torch.tensor(log_probs)

    # Atualizar política
    for _ in range(5):  # 5 épocas de otimização
        # Obter novas probabilidades e valores
        new_action_probs, values = model(states_tensor)
        values = values.squeeze()
        
        # Selecionar probabilidades das ações tomadas
        new_log_probs = torch.log(new_action_probs.gather(1, actions_tensor.unsqueeze(1))).squeeze()
        
        # Calcular razões e vantagens
        ratios = torch.exp(new_log_probs - old_log_probs)
        advantages = returns - values.detach()
        
        # Calcular perdas do PPO
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1-clip_epsilon, 1+clip_epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = nn.MSELoss()(values, returns)
        entropy = -(new_action_probs * torch.log(new_action_probs + 1e-10)).sum(dim=1).mean()
        
        # Perda total
        loss = actor_loss + c1 * critic_loss - c2 * entropy
        
        # Otimizar
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if episode % 10 == 0:
        mean_reward = np.mean(episode_rewards[-10:] if len(episode_rewards) >= 10 else episode_rewards)
        print(f'Episódio {episode} - Recompensa média: {mean_reward:.2f}')

env.close()

# Métricas
print("\nMétricas PPO:")
recompensa_cumulativa = np.sum(episode_rewards)
print(f'Recompensa Cumulativa: {recompensa_cumulativa}')
taxa_convergencia = np.mean(episode_rewards[-10:])
print(f'Taxa de Convergência: {taxa_convergencia}')
print(f'Taxa de Aprendizado: {0.002:.2%}')

# Visualizar resultados
plt.figure(figsize=(10, 5))
plt.plot(episode_rewards)
plt.xlabel('Episódios')
plt.ylabel('Recompensa Total')
plt.title('Evolução das Recompensas no PPO')
plt.show()