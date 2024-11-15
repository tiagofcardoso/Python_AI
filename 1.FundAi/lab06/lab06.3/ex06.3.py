import pandas as pd
import matplotlib.pyplot as plt

# Carregar os dados
df = pd.read_csv('dados_demanda_energia.csv', parse_dates=['Data'], index_col='Data')

# Visualizar as primeiras linhas e a série temporal
print(df.head())
df.plot(figsize=(14, 6), title='Demanda de Energia')
plt.show()