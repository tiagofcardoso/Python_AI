#Crie um programa em Python que utilize uma fórmula matemática simples para estimar a probabilidade de um cliente pagar um empréstimo, com base na sua idade e rendimento mensal. A fórmula deve ser a seguinte:
#probabilidade_pagamento = 0.01 * idade + 0.0005 * rendimento - 0.5
#O programa deve pedir ao utilizador a sua idade e o seu rendimento mensal, calcular a probabilidade de pagamento usando a fórmula e imprimir o resultado em percentagem.

# Solicita a idade do utilizador
idade = int(input("Por favor, insira a sua idade: "))

# Solicita o rendimento mensal do utilizador
rendimento_mensal = float(input("Por favor, insira o seu rendimento mensal em euros: "))

# Calcula a probabilidade de pagamento
probabilidade_pagamento = 0.01 * idade + 0.0005 * rendimento_mensal - 0.5

# Imprime a probabilidade de pagamento
print(f"A probabilidade de pagamento é de {probabilidade_pagamento:.2%}.")

