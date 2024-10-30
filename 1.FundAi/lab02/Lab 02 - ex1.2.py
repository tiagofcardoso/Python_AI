# Representação simbólica

L1 = "cidade"
L2 = "praia"
L3 = "montanha"

C1 = "sol"
C2 = "frio"
C3 = "neve"

I1 = "historia"
I2 = "aventura"
I3 = "relaxamento"

D1 = "curta"
D2 = "moderada"
D3 = "longa"

ACAO_DEFEITO = "Nenhuma atividade recomendada para esta combinação de fatores."
A1 = "Visitar museus e fazer um passeio guiado pela cidade."
A2 = "Visitar museus, fazer um passeio guiado e explorar bairros históricos."
A3 = "Fazer um passeio de bicicleta e uma caminhada urbana."
A4 = "Ficar na praia, fazer yoga ao ar livre e relaxar no spa."
A5 = "Fazer esqui, snowboarding e explorar trilhas de neve."

# Definir os fatores
tipo_destino = input("Tipo de destino? :")  # Pode ser "cidade", "praia" ou "montanha"
preferencia_climatica = input(
    "Preferencia Climatica? :"
)  # Pode ser "sol", "frio" ou "neve"
interesse_principal = input(
    "Interesse Principal? :"
)  # Pode ser "historia", "aventura" ou "relaxamento"
duracao_estadia = input(
    "Tempo de estadia? :"
)  # Pode ser "curta", "moderada" ou "longa"

# Mapeamento das combinações de fatores para atividades
atividades = {
    (L1, C1, I1, D1): A1,
    (L1, C1, I1, D2): A2,
    (L1, C1, I2, D1): A3,
    (L2, C1, I3, D3): A4,
    (L3, C3, I2, D3): A5,
}

# Decidir atividades com base nos fatores
decisao = atividades.get(
    (tipo_destino, preferencia_climatica, interesse_principal, duracao_estadia),
    ACAO_DEFEITO,
)

print(decisao)
