# Representação simbolica

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
tipo_destino = str(
    input("Tipo de destino? :")
)  # Pode ser "cidade", "praia" ou "montanha"
preferencia_climatica = str(
    input("Preferencia Climatica? :")
)  # Pode ser "sol", "frio" ou "neve"
interesse_principal = str(
    input("Interesse Principal? :")
)  # Pode ser "historia", "aventura" ou "relaxamento"
duracao_estadia = str(
    input("Tempo de estadia? :")
)  # Pode ser "curta", "moderada" ou "longa"

decisao = ACAO_DEFEITO  # Decisão padrão para o caso de não haver uma combinação de fatores válida SAFE STATE

# Decidir atividades com base nos fatores
if (
    tipo_destino == L1
    and preferencia_climatica == C1
    and interesse_principal == I1
    and duracao_estadia == D1
):
    decisao = A1
elif (
    tipo_destino == L1
    and preferencia_climatica == C1
    and interesse_principal == I1
    and duracao_estadia == D2
):
    decisao = A2
elif (
    tipo_destino == L1
    and preferencia_climatica == C1
    and interesse_principal == I2
    and duracao_estadia == D1
):
    decisao = A3
elif (
    tipo_destino == L2
    and preferencia_climatica == C1
    and interesse_principal == I3
    and duracao_estadia == D3
):
    decisao = A4
elif (
    tipo_destino == L3
    and preferencia_climatica == C3
    and interesse_principal == I2
    and duracao_estadia == D3
):
    decisao = A5
print(decisao)
