def validar_nif(nif):
    """
    Valida um Número de Identificação Fiscal (NIF) português.
    
    Parâmetros:
    nif (str): O NIF a ser validado.
    
    Retorna:
    bool: True se o NIF for válido, False caso contrário.
    """
    if len(nif) != 9 or not nif.isdigit():
        return False

    # Verifica se o primeiro dígito é válido
    if nif[0] not in '125689':
        return False

    # Calcula o dígito de controle
    total = sum(int(digito) * (9 - idx) for idx, digito in enumerate(nif[:8]))
    digito_controle = 11 - (total % 11)
    if digito_controle >= 10:
        digito_controle = 0

    return digito_controle == int(nif[8])

# Entrada pelo teclado
nif = input("Por favor, insira o seu NIF: ")
if validar_nif(nif):
    print(f"O NIF {nif} é válido.")
else:
    print(f"O NIF {nif} é inválido.")