# Solicita a idade do utilizador
idade = int(input("Por favor, insira a sua idade: "))

# Solicita o rendimento mensal do utilizador
rendimento_mensal = float(input("Por favor, insira o seu rendimento mensal em euros: "))

# Verifica se o utilizador é elegível para o empréstimo
if idade >= 18 and rendimento_mensal >= 1000:
    print("Parabéns! Você é elegível para o empréstimo.")
else:
    print("Desculpe, você não é elegível para o empréstimo.")