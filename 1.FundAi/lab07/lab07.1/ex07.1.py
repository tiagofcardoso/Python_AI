# Importações iniciais
from scipy.spatial.distance import cdist
import numpy as np
from sklearn.metrics import silhouette_score, adjusted_rand_score, v_measure_score
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

# --- Ex 1: Carregar os Dados do Ficheiro CSV ---
# Carregar os dados do CSV
df = pd.read_csv('~/AIEngineer/1.FundAi/lab07/lab07.1/vinho.csv')
print("Primeiros exemplos do dataset:\n", df.head())

# Separar as features e a coluna de classe
X = df.drop(columns=['classe'])
y_true = df['classe'].values

# Normalizar os dados para clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduzir a dimensionalidade para 2D usando PCA (apenas para visualização)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Visualizar o dataset em 2D
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap='viridis', s=30)
plt.title("Dataset de Vinhos (Classes Reais)")
plt.savefig('vinho_dataset.png')
plt.show()
