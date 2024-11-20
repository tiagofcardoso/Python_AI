import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np

# Ex 1: Carregar os Dados do Ficheiro CSV
df = pd.read_csv('/home/tiagocardoso/AIEngineer/1.FundAi/lab07/lab07.2/vinho.csv')
print("Primeiros exemplos do dataset:\n", df.head())

# Separar as features e a coluna de classe
X = df.drop(columns=['classe'])
y = df['classe']

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Ex 2: Aplicar PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

#visualizar os dados
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.title('Redução PCA')
plt.show()

# Calcular a distancia de recosntrução do pca
X_pca_reconstructed = pca.inverse_transform(X_pca)
reconstruction_error = np.mean(np.sum((X_scaled - X_pca_reconstructed) ** 2, axis=1))   
print("Erro de reconstrução PCA: ", reconstruction_error)

# Ex 3: Aplicar Kernel PCA
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1)
X_kpca = kpca.fit_transform(X_scaled)
