def neuronio(entradas, pesos, bias):
    ativacao = sum([entrada * peso for entrada, peso in zip(entradas, pesos)]) + bias
    return 1 if ativacao >= 0 else 0


# Teste
idade = int(input("Por favor, insira a sua idade: "))
rendimento_mensal = float(input("Por favor, insira o seu rendimento mensal em euros: "))

entradas = [idade, rendimento_mensal]
pesos = [0.01, 0.0005]
bias = -0.5

resultado = neuronio(entradas, pesos, bias)
print(
    "Neuronio: Voce é elegível para o empréstimo."
    if resultado == 1
    else "Voce não é elegível para o empréstimo."
)

