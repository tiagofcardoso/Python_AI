import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, f1_score, classification_report
import joblib

# 1. Carregar e preparar os dados
def carregar_dados(arquivo):
    df = pd.read_csv(arquivo)
    X = df[['idade', 'renda', 'pontuacao_credito', 'gasto_anual']]
    y = df['categoria_cliente']
    return X, y

# 2. Dividir os dados
def dividir_dados(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

# 3. Treinar e avaliar modelos
def treinar_avaliar_modelo(modelo, X_train, X_val, y_train, y_val, nome_modelo):
    # Treinar modelo
    modelo.fit(X_train, y_train)   

    # Fazer previsões
    y_val_pred = modelo.predict(X_val)  

    # Calcular métricas
    acc = accuracy_score(y_val, y_val_pred)
    prec = precision_score(y_val, y_val_pred, average='weighted')
    f1 = f1_score(y_val, y_val_pred, average='weighted')
    
    return {
        'modelo': modelo,
        'nome': nome_modelo,
        'acuracia': acc,
        'precisao': prec,
        'f1_score': f1
    }

# 4. Função principal
def main():
    # Carregar dados
    print("Carregar dados...")
    X, y = carregar_dados(
        '/home/tiagocardoso/AIEngineer/1.FundAi/lab06/lab06.2/dados_clientes.csv')

    # Dividir dados
    print("Dividir dados em conjuntos de treino, validação e teste...")
    X_train, X_val, X_test, y_train, y_val, y_test = dividir_dados(X, y)

        # Definir modelos para teste
    modelos = [
        (KNeighborsClassifier(n_neighbors=5), 'KNN'),
        (LogisticRegression(max_iter=1000), 'Regressao Logistica'),
        (SVC(kernel='rbf'), 'SVC')
    ]

    # Treinar e avaliar cada modelo
    resultados = []
    for modelo, nome in modelos:
        print(f"Processando {nome}...")
        resultado = treinar_avaliar_modelo(modelo, X_train, X_val, y_train, y_val, nome)
        resultados.append(resultado)        

    # Encontrar o melhor modelo
    melhor_modelo = max(resultados, key=lambda x: x['acuracia'])

    print("\n" + "=" * 50)
    print(f"\nMelhor modelo: {melhor_modelo['nome']}")
    print(f"Acurácia: {melhor_modelo['acuracia']:.4f}")

    # Salvar melhor modelo
    print("\nSalvar melhor modelo...")
    joblib.dump(melhor_modelo['modelo'], f"melhor_modelo_{melhor_modelo['nome']}.pkl")
    print("Modelo salvo com sucesso!")

        # Avaliação final no conjunto de teste
    y_test_pred = melhor_modelo['modelo'].predict(X_test)
    print("\nRelatório de classificação no conjunto de teste:")
    print(classification_report(y_test, y_test_pred))
    
if __name__ == '__main__':
        main()