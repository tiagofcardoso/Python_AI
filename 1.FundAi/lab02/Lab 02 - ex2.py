import math
import pygame

# Configurações da tela
LARGURA, ALTURA = 800, 600

# Posições dos locais
POSICOES = {
    "Base": (400, 550),
    "A": (200, 200),
    "B": (600, 200),
    "C": (200, 400),
    "D": (600, 400),
}

# Configurações do obstáculo
OBSTACULO_LARGURA = 20
OBSTACULO_ALTURA = 200
OBSTACULO_VELOCIDADE = 2


def representar_e_inicializar_o_sistema():
    return {
        "estado_drone": "Descolagem",
        "posicao_atual": "Base",
        "posicao_drone": POSICOES["Base"],
        "destino": POSICOES["A"],
        "proximo_destino": "A",
        "movimento_completo": False,
        "obstaculo_y": (ALTURA - OBSTACULO_ALTURA) // 2,
        "obstaculo_direcao": 1,  # 1 para baixo, -1 para cima
        "rota_alternativa": None,
        "evitando_obstaculo": False,
    }


def regra_atualizar_estado_voo(sistema):
    sistema = regra_mover_drone(sistema)

    if sistema["movimento_completo"] and sistema["estado_drone"] == "Descolagem":
        sistema["estado_drone"] = "Em Rota"
    if (
        sistema["movimento_completo"]
        and sistema["estado_drone"] == "Em Rota"
        and not sistema["evitando_obstaculo"]
    ):
        sistema["posicao_atual"] = sistema["proximo_destino"]
        proxima_posicao = acao_obter_proxima_posicao(sistema["posicao_atual"])
        sistema["proximo_destino"] = proxima_posicao
        sistema["destino"] = POSICOES[proxima_posicao]
        sistema["movimento_completo"] = False
        sistema["rota_alternativa"] = None

    # Atualizar posição do obstáculo
    sistema["obstaculo_y"] += OBSTACULO_VELOCIDADE * sistema["obstaculo_direcao"]
    if (
        sistema["obstaculo_y"] <= 0
        or sistema["obstaculo_y"] + OBSTACULO_ALTURA >= ALTURA
    ):
        sistema["obstaculo_direcao"] *= -1

    return sistema


def regra_mover_drone(sistema):
    velocidade = 5
    x1, y1 = sistema["posicao_drone"]
    x2, y2 = sistema["destino"]

    dx = x2 - x1
    dy = y2 - y1
    distancia = math.sqrt(dx**2 + dy**2)

    if distancia >= velocidade:
        fator = velocidade / distancia
        novo_x = x1 + dx * fator
        novo_y = y1 + dy * fator

        sistema["posicao_drone"] = (novo_x, novo_y)

    if distancia < velocidade:
        sistema["posicao_drone"] = sistema["destino"]
        sistema["movimento_completo"] = True
        sistema["evitando_obstaculo"] = False

    return sistema


def acao_obter_proxima_posicao(posicao_atual):
    ordem = ["A", "B", "C", "D", "Base"]
    idx = ordem.index(posicao_atual)
    return ordem[(idx + 1) % len(ordem)]


def simular_sistema():
    sistema = representar_e_inicializar_o_sistema()
    for _ in range(500):  # Simular 500 passos
        sistema = regra_atualizar_estado_voo(sistema)
        print(sistema['estado_drone'], sistema['posicao_drone'])


if __name__ == "__main__":
    simular_sistema()
