# Feito na mão
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Import dataset

url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)

print("Valores Ausentes por Coluna:")
print(df.isnull().sum())

print()

# Treatment of missing data
df['Age'].fillna(df['Age'].median(), inplace=True)
df.drop(['Cabin'], axis=1, inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Conversão de dados categóricos para variáveis dummy
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Remover colunas que não serão utilizadas
df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)


y = df['Survived']
x = df.drop('Survived', axis=1)

# standardize the data 2 variables
# scaler = StandardScaler()
# x[['Age', 'Fare']] = scaler.fit_transform(x[['Age', 'Fare']])

# Dividir em treino e teste (80% treino, 20% teste)
x_traing_temp, x_test, y_train_temp, y_test_split = train_test_split(
    x, y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(
    x_traing_temp, y_train_temp, test_size=0.2, random_state=42)

# Treinar o modelo
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# Avaliar o modelo
print('Avaliação com conjunto de validação')
y_val_pred = model.predict(x_val)
print(f"Acurácia: {accuracy_score(y_val, y_val_pred):.2f}")
print(classification_report(y_val, y_val_pred))


# avaliação final com conjunto de teste
print('Avaliação final com conjunto de teste')
y_test_pred = model.predict(x_test)
print(f"Acurácia: {accuracy_score(y_test_split, y_test_pred):.2f}")
print(classification_report(y_test_split, y_test_pred))


# Exemplo de Previsão de um novo caso
print('Exemplo de Previsão de um novo caso')
# Exemplo de Previsão para Novo Caso
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

sobreviveu = model.predict(novo_caso)
print(f'O modelo prevê que o passageiro sobreviveu? \n {
      'Nasceu de Novo! ' if sobreviveu == 0 else "Morreu"}')

# Coeficientes do Moelo
print()
print('Coeficientes do Modelo')
coeficientes = pd.DataFrame(
    model.coef_[0], index=x.columns)
print(coeficientes)

# Grafico de Coeficientes
plt.figure(figsize=(30, 10))
sns.barplot(x=coeficientes.values.flatten(), y=coeficientes.index)
plt.title('Coeficientes do Modelo de Regressão Logística')
plt.xlabel('Coeficiente')
plt.ylabel('Variável')
plt.show()

# Gráfico de Acurácia
accuracy_scores = {
    'Validação': accuracy_score(y_val, y_val_pred),
    'Teste': accuracy_score(y_test_split, y_test_pred)
}

plt.figure(figsize=(20, 8))
sns.barplot(x=list(accuracy_scores.keys()), y=list(accuracy_scores.values()))
plt.title('Acurácia do Modelo')
plt.xlabel('Conjunto de Dados')
plt.ylabel('Acurácia')
plt.ylim(0, 1)
plt.show()
