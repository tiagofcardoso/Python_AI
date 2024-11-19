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
