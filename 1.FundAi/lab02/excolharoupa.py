def escolher_roupa(temperatura):
    if temperatura < 15:
        return "Vestir um casaco"
    elif 15 <= temperatura <= 25:
        return "Vestir uma camisola"
    else:
        return "Vestir uma t-shirt"

# Exemplo de uso
temperatura = int(input("Qual temperatura atual? : "))
acao = escolher_roupa(temperatura)
print(f"Temperatura: {temperatura}°C")
print(f"Ação: {acao}")