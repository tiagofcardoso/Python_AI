import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados
df = pd.read_csv(
    '/home/tiagocardoso/AIEngineer/1.FundAi/lab06/lab06.4/fidelidade_clientes.csv')

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

#ex 1: Modelo de Árvore de Decisão

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# Separar variáveis independentes e dependente
X = df.drop(columns=['Fidelidade'])  # Variáveis independentes
y = df['Fidelidade']                 # Variável dependente (classe)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo de Árvore de Decisão
modelo_decision_tree = DecisionTreeClassifier(random_state=42)
modelo_decision_tree.fit(X_train, y_train)

# Avaliar o modelo
y_pred_dt = modelo_decision_tree.predict(X_test)
print("Acurácia (Árvore de Decisão):", accuracy_score(y_test, y_pred_dt))
print("AUC (Árvore de Decisão):", roc_auc_score(y_test, modelo_decision_tree.predict_proba(X_test)[:, 1]))

#ex 2: Modelo Random Forest

from sklearn.ensemble import RandomForestClassifier

# Treinar o modelo Random Forest
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_rf.fit(X_train, y_train)

# Avaliar o modelo
y_pred_rf = modelo_rf.predict(X_test)
print("Acurácia (Random Forest):", accuracy_score(y_test, y_pred_rf))
print("AUC (Random Forest):", roc_auc_score(y_test, modelo_rf.predict_proba(X_test)[:, 1]))

#ex 3: Modelo XGBoost
 
import xgboost as xgb

# Converter os dados para o formato DMatrix do XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Definir parâmetros e treinar o modelo
param = {'max_depth': 3, 'eta': 0.1, 'objective': 'binary:logistic'}
num_round = 100
modelo_xgb = xgb.train(param, dtrain, num_round)

# Avaliar o modelo
y_pred_xgb = modelo_xgb.predict(dtest)
y_pred_xgb_binary = [1 if prob > 0.5 else 0 for prob in y_pred_xgb]
print("Acurácia (XGBoost):", accuracy_score(y_test, y_pred_xgb_binary))
print("AUC (XGBoost):", roc_auc_score(y_test, y_pred_xgb))

#ex 4: Modelo CatBoost

from catboost import CatBoostClassifier

# Treinar o modelo CatBoost
modelo_cb = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=3, verbose=0)
modelo_cb.fit(X_train, y_train)

# Avaliar o modelo
y_pred_cb = modelo_cb.predict(X_test)
print("Acurácia (CatBoost):", accuracy_score(y_test, y_pred_cb))
print("AUC (CatBoost):", roc_auc_score(y_test, modelo_cb.predict_proba(X_test)[:, 1]))

#ex 5: Comparação de Modelos

resultados = {
    'Modelo': ['Árvore de Decisão', 'Random Forest', 'XGBoost', 'CatBoost'],
    'Acurácia': [
        accuracy_score(y_test, y_pred_dt),
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_xgb_binary),
        accuracy_score(y_test, y_pred_cb)
    ],
    'AUC': [
        roc_auc_score(y_test, modelo_decision_tree.predict_proba(X_test)[:, 1]),
        roc_auc_score(y_test, modelo_rf.predict_proba(X_test)[:, 1]),
        roc_auc_score(y_test, y_pred_xgb),
        roc_auc_score(y_test, modelo_cb.predict_proba(X_test)[:, 1])
    ]
}

resultados_df = pd.DataFrame(resultados)
print(resultados_df)