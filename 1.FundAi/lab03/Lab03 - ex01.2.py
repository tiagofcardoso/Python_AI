import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import pickle

# Carregar o dataset Titanic diretamente da internet e fazer uma inspeção inicial.
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)

print(df.head())
print(df.info())
print(df.describe())

# Tratamento de dados
print("Valores Ausentes por Coluna:")
print(df.isnull().sum())

# Preencher os valores ausentes da coluna 'Age' com a mediana da coluna
df['Age'].fillna(df['Age'].median(), inplace=True)

# Remover a coluna 'Cabin' devido a alta quantidade de valores ausentes
df.drop(['Cabin'], axis=1, inplace=True)

# Preencher os valores ausentes em 'Embarked' com o valor mais frequente
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Converter "Sex" e "Embarked" para variáveis dummy
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Remover colunas que não serão utilizadas
df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

# Dividir dados em X e y
X = df.drop('Survived', axis=1)
y = df['Survived']

# Dividir em treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar o modelo de regressão logística
modelo = LogisticRegression(max_iter=200)

# Treinar o modelo
modelo.fit(X_train, y_train)

# Previsões no conjunto de teste
y_test_pred = modelo.predict(X_test)

# Avaliar precisão e exibir o relatório de classificação
print(f"Precisão no Conjunto de Teste: {accuracy_score(y_test, y_test_pred):.2f}")
print("Relatório de Classificação (Teste):")
print(classification_report(y_test, y_test_pred))

# Criar um novo caso para previsão
novo_caso = pd.DataFrame({
    'Pclass': [2],
    'Age': [28],
    'SibSp': [0],
    'Parch': [0],
    'Fare': [20],
    'Sex_male': [1],
    'Embarked_Q': [0],
    'Embarked_S': [1]
})

# Fazer a previsão
sobreviveu = modelo.predict(novo_caso)
print(f"Previsão de Sobrevivência do Novo Passageiro: {'Sobreviveu' if sobreviveu[0] == 1 else 'Não Sobreviveu'}")

# Coeficientes do modelo
coeficientes = pd.DataFrame(modelo.coef_, columns=X.columns)
print("Coeficientes do Modelo:")
print(coeficientes.T)

# Guardar os nomes das colunas de preditores
input_columns = X.columns.tolist()
pickle.dump(input_columns, open('input_columns.sav', 'wb'))
