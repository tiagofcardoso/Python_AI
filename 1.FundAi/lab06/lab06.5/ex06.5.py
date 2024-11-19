from transformers import pipeline

modelo = pipeline("sentiment-analysis")
print("Modelo carregado com sucesso!")

dataset = [
    {"frase": "Este filme foi maravilhoso, gostei muito!", "sentimento_esperado": "POSITIVE"},
    {"frase": "O produto não correspondeu às minhas expectativas, fiquei dececionado.", "sentimento_esperado": "NEGATIVE"},
    {"frase": "Que experiência fantástica! Adorei tudo.", "sentimento_esperado": "POSITIVE"},
    {"frase": "Foi uma perda de tempo, não recomendo.", "sentimento_esperado": "NEGATIVE"}
]

# Função para classificar o sentimento de uma frase
def classificar_sentimento(modelo, frase):
    # Realizar a classificação
    resultado = modelo(frase)[0]
    sentimento = resultado['label']
    
    # Mapear para "Positivo" ou "Negativo"
    if sentimento == "POSITIVE":
        return "POSITIVE"
    elif sentimento == "NEGATIVE":
        return "NEGATIVE"
    else:
        return "Indeterminado"
    

# Avaliação do modelo
corretos = 0
for item in dataset:
    frase = item["frase"]
    sentimento_esperado = item["sentimento_esperado"]

    sentimento_predito = classificar_sentimento(modelo, frase)

    # Verificar se o sentimento predito corresponde ao esperado
    resultado = "Correto" if sentimento_predito == sentimento_esperado else "Incorreto"
    print(f"Frase: {frase}")
    print(f"Sentimento Esperado: {sentimento_esperado}")
    print(f"Sentimento Predito: {sentimento_predito} ({resultado})\n")
    
    if sentimento_predito == sentimento_esperado:
        corretos += 1

# Calcular a precisão
precisao = (corretos / len(dataset)) * 100
print(f"Precisão do modelo: {precisao}%")