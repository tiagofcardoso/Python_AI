import pandas as pd 
import numpy as np
import pickle


# Carregar o dataset
df = pd.read_csv("dataset.csv")

# Verificar a estrutura dos dados
df.shape
df.head()

# Verificar se há valores em falta
df.isnull().values.any()

# Converter a variável LABEL_TARGET para valores numéricos
df["LABEL_TARGET"] = df["LABEL_TARGET"].astype(int)

# Criar subconjuntos com amostragem aleatória
df_sample_30 = df.sample(frac=0.3)

# Conjunto de Teste e Validação
df_test = df_sample_30.sample(frac=0.5)
df_valid = df_sample_30.drop(df_test.index)

# Conjunto de Treino
df_train = df.drop(df_sample_30.index)

# Separar as classes positivas e negativas
df_train_pos = df_train[df_train.LABEL_TARGET == 1]
df_train_neg = df_train[df_train.LABEL_TARGET == 0]

# Encontrar o tamanho mínimo entre as duas classes
min_value = min(len(df_train_pos), len(df_train_neg))

# Subamostragem para balancear as classes
df_train_final = pd.concat([df_train_pos.sample(n=min_value),
df_train_neg.sample(n=min_value)])

# Verificar o balanceamento das classes
df_train_final.LABEL_TARGET.value_counts()

# Guardar os datasets em CSV
df_train_final.to_csv('train_final_data.csv', index=False)
df_valid.to_csv('validation_data.csv', index=False)
df_test.to_csv('test_data.csv', index=False)

# Guardar os nomes das colunas de preditores
input_columns = df.columns.tolist()
pickle.dump(input_columns, open('input_columns.sav', 'wb'))