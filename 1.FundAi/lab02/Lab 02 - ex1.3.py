# Representação Simbólica
sys = {
    "L1": "cidade",
    "L2": "praia",
    "L3": "montanha",
    "C1": "sol",
    "C2": "frio",
    "C3": "neve",
    "I1": "historia",
    "I2": "aventura",
    "I3": "relaxamento",
    "D1": "curta",
    "D2": "moderada",
    "D3": "longa",
}
acoes = {
    "ACAO_DEFEITO": "Nenhuma atividade recomendada para esta combinação de fatores.",
    "ACAO1": "Visitar museus e fazer um passeio guiado pela cidade.",
    "ACAO2": "Visitar museus, fazer um passeio guiado e explorar bairros históricos.",
    "ACAO3": "Fazer um passeio de bicicleta e uma caminhada urbana.",
    "ACAO4": "Ficar na praia, fazer yoga ao ar livre e relaxar no spa.",
    "ACAO5": "Fazer esqui, snowboarding e explorar trilhas de neve.",
}

# Definir os fatores
L = "cidade"  # Pode ser "cidade", "praia" ou "montanha"
C = "sol"  # Pode ser "sol", "frio" ou "neve"
I = "historia"  # Pode ser "historia", "aventura" ou "relaxamento"
D = "moderada"  # Pode ser "curta", "moderada" ou "longa"

decisao = acoes["ACAO_DEFEITO"]  # Safe State 

# Decidir atividades com base nos fatores
def motor_decisao(L, C, I, D):
    if L == sys["L1"] and C == sys["C1"] and I == sys["I1"] and D == sys["D1"]:
        decisao = acoes["ACAO1"]
    elif L == sys["L1"] and C == sys["C1"] and I == sys["I1"] and D == sys["D2"]:
        decisao = acoes["ACAO2"]
    elif L == sys["L1"] and C == sys["C1"] and I == sys["I2"] and D == sys["D1"]:
        decisao = acoes["ACAO3"]
    elif L == sys["L2"] and C == sys["C1"] and I == sys["I3"] and D == sys["D3"]:
        decisao = acoes["ACAO4"]
    elif L == sys["L3"] and C == sys["C3"] and I == sys["I2"] and D == sys["D3"]:
        decisao = acoes["ACAO5"]


print(decisao)

# Representação Simbólica
sys = {
    "L1": "cidade",
    "L2": "praia",
    "L3": "montanha",
    "C1": "sol",
    "C2": "frio",
    "C3": "neve",
    "I1": "historia",
    "I2": "aventura",
    "I3": "relaxamento",
    "D1": "curta",
    "D2": "moderada",
    "D3": "longa",
}

Acao = {
    "ACAO_DEFEITO": "Nenhuma atividade recomendada para esta combinação de fatores.",
    "ACAO1": "Visitar museus e fazer um passeio guiado pela cidade.",
    "ACAO2": "Visitar museus, fazer um passeio guiado e explorar bairros históricos.",
    "ACAO3": "Fazer um passeio de bicicleta e uma caminhada urbana.",
    "ACAO4": "Ficar na praia, fazer yoga ao ar livre e relaxar no spa.",
    "ACAO5": "Fazer esqui, snowboarding e explorar trilhas de neve.",
}


# Decidir atividades com base nos fatores
def motor_decisao(L, C, I, D):
    decisao = Acao["ACAO_DEFEITO"]  # Safe State

    if L == sys["L1"] and C == sys["C1"] and I == sys["I1"] and D == sys["D1"]:
        decisao = Acao["ACAO1"]
    elif L == sys["L1"] and C == sys["C1"] and I == sys["I1"] and D == sys["D2"]:
        decisao = Acao["ACAO2"]
    elif L == sys["L1"] and C == sys["C1"] and I == sys["I2"] and D == sys["D1"]:
        decisao = Acao["ACAO3"]
    elif L == sys["L2"] and C == sys["C1"] and I == sys["I3"] and D == sys["D3"]:
        decisao = Acao["ACAO4"]
    elif L == sys["L3"] and C == sys["C3"] and I == sys["I2"] and D == sys["D3"]:
        decisao = Acao["ACAO5"]
    return decisao


# Definir os fatores
x = ("cidade", "sol", "historia", "moderada")
y = motor_decisao(x[0], x[1], x[2], x[3])
print(y)
