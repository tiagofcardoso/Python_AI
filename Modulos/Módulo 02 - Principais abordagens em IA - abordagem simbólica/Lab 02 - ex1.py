# Definir os fatores
tipo_destino = "cidade"  # Pode ser "cidade", "praia" ou "montanha"
preferencia_climatica = "sol"  # Pode ser "sol", "frio" ou "neve"
interesse_principal = "historia"  # Pode ser "historia", "aventura" ou "relaxamento"
duracao_estadia = "moderada"  # Pode ser "curta", "moderada" ou "longa"

# Decidir atividades com base nos fatores
if tipo_destino == "cidade" and preferencia_climatica == "sol" and interesse_principal == "historia" and duracao_estadia == "curta":
    print("Visitar museus e fazer um passeio guiado pela cidade.")
elif tipo_destino == "cidade" and preferencia_climatica == "sol" and interesse_principal == "historia" and duracao_estadia == "moderada":
    print("Visitar museus, fazer um passeio guiado e explorar bairros históricos.")
elif tipo_destino == "cidade" and preferencia_climatica == "sol" and interesse_principal == "aventura" and duracao_estadia == "curta":
    print("Fazer um passeio de bicicleta e uma caminhada urbana.")
elif tipo_destino == "praia" and preferencia_climatica == "sol" and interesse_principal == "relaxamento" and duracao_estadia == "longa":
    print("Ficar na praia, fazer yoga ao ar livre e relaxar no spa.")
elif tipo_destino == "montanha" and preferencia_climatica == "neve" and interesse_principal == "aventura" and duracao_estadia == "longa":
    print("Fazer esqui, snowboarding e explorar trilhas de neve.")
else:
    print("Nenhuma atividade recomendada para esta combinação de fatores.")
