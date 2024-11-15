import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados
df = pd.read_csv('fidelidade_clientes.csv')

# Visualizar as primeiras linhas
print(df.head())

# Verificar a distribuição da variável alvo (ex: "Fidelidade")
sns.countplot(x='Fidelidade', data=df)
plt.title('Distribuição da Fidelidade dos Clientes')
plt.show()

# Analisar as correlações entre variáveis
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Mapa de Correlação')
plt.show()
