import pygame
import sys
import math

# Initialize Pygame
pygame.init()

# Screen settings
LARGURA, ALTURA = 800, 600
tela = pygame.display.set_mode((LARGURA, ALTURA))
pygame.display.set_caption("Simulação de Drone - Ronda com Obstáculo")

# Colors
PRETO = (0, 0, 0)
BRANCO = (255, 255, 255)
VERMELHO = (255, 0, 0)
VERDE = (0, 255, 0)
AZUL = (0, 0, 255)
AMARELO = (255, 255, 0)

# Positions of locations
POSICOES = {
    "Base": (400, 550),
    "A": (200, 200),
    "B": (600, 200),
    "C": (200, 400),
    "D": (600, 400),
}

# Obstacle settings
OBSTACULO_LARGURA = 20
OBSTACULO_ALTURA = 200
OBSTACULO_VELOCIDADE = 2

# Drone settings
VELOCIDADE_DRONE = 5


def representar_e_inicializar_o_sistema():
    """Initialize the system state."""
    return {
        "estado_drone": "Descolagem",
        "posicao_atual": "Base",
        "posicao_drone": POSICOES["Base"],
        "destino": POSICOES["A"],
        "proximo_destino": "A",
        "movimento_completo": False,
        "obstaculo_y": (ALTURA - OBSTACULO_ALTURA) // 2,
        "obstaculo_direcao": 1,  # 1 for down, -1 for up
        "rota_alternativa": None,
        "evitando_obstaculo": False,
    }


def regra_atualizar_estado_voo(sistema):
    """Update the flight state of the drone."""
    sistema = regra_mover_drone(sistema)
    sistema = atualizar_estado_drone(sistema)
    sistema = atualizar_posicao_obstaculo(sistema)
    return sistema


def atualizar_estado_drone(sistema):
    """Update the drone's state based on its movement."""
    if sistema["movimento_completo"] and sistema["estado_drone"] == "Descolagem":
        sistema["estado_drone"] = "Em Rota"
    if (
        sistema["movimento_completo"]
        and sistema["estado_drone"] == "Em Rota"
        and not sistema["evitando_obstaculo"]
    ):
        sistema["posicao_atual"] = sistema["proximo_destino"]
        proxima_posicao = ação_obter_proxima_posicao(sistema["posicao_atual"])
        sistema["proximo_destino"] = proxima_posicao
        sistema["destino"] = POSICOES[proxima_posicao]
        sistema["movimento_completo"] = False
        sistema["rota_alternativa"] = None
    return sistema


def atualizar_posicao_obstaculo(sistema):
    """Update the position of the obstacle."""
    sistema["obstaculo_y"] += OBSTACULO_VELOCIDADE * sistema["obstaculo_direcao"]
    if (
        sistema["obstaculo_y"] <= 0
        or sistema["obstaculo_y"] + OBSTACULO_ALTURA >= ALTURA
    ):
        sistema["obstaculo_direcao"] *= -1
    return sistema


def regra_calcular_rota_alternativa(sistema, x1, y1, x2, y2):
    """Calculate an alternative route to avoid the obstacle."""
    obstaculo_x = LARGURA // 2
    obstaculo_y = sistema["obstaculo_y"]

    # Drone is to the right of the obstacle
    ponto_desvio = (
        obstaculo_x + OBSTACULO_LARGURA + 10,
        min(obstaculo_y - 20, max(x1, x2)),
    )

    if x1 < obstaculo_x:
        ponto_desvio = (
            obstaculo_x + OBSTACULO_LARGURA + 10,
            min(obstaculo_y - 20, max(x1, x2)),
        )

    return ponto_desvio


def regra_verifica_colisao(x1, y1, x2, y2, obstaculo_y):
    """Check for collision with the obstacle."""
    obstaculo_x = LARGURA // 2
    if min(x1, x2) <= obstaculo_x <= max(x1, x2):
        if x1 != x2:
            t = (obstaculo_x - x1) / (x2 - x1)
            y_intercesao = y1 + t * (y2 - y1)
            if obstaculo_y <= y_intercesao <= obstaculo_y + OBSTACULO_ALTURA:
                return True
    return False


def regra_mover_drone(sistema):
    """Move the drone towards its destination."""
    x1, y1 = sistema["posicao_drone"]
    x2, y2 = sistema["destino"]

    if sistema["rota_alternativa"]:
        x2, y2 = sistema["rota_alternativa"]

    dx = x2 - x1
    dy = y2 - y1
    distancia = math.sqrt(dx**2 + dy**2)

    if distancia >= VELOCIDADE_DRONE:
        fator = VELOCIDADE_DRONE / distancia
        novo_x = x1 + dx * fator
        novo_y = y1 + dy * fator

        sistema["posicao_drone"] = (novo_x, novo_y)

        if regra_verifica_colisao(x1, y1, novo_x, novo_y, sistema["obstaculo_y"]):
            if not sistema["rota_alternativa"]:
                sistema["rota_alternativa"] = regra_calcular_rota_alternativa(
                    sistema, x1, y1, x2, y2
                )
                sistema["evitando_obstaculo"] = True

    if distancia < VELOCIDADE_DRONE:
        if sistema["rota_alternativa"]:
            sistema["posicao_drone"] = sistema["rota_alternativa"]
            sistema["rota_alternativa"] = None
        sistema["posicao_drone"] = sistema["destino"]
        sistema["movimento_completo"] = True
        sistema["evitando_obstaculo"] = False

    return sistema


def ação_obter_proxima_posicao(posicao_atual):
    """Get the next position in the patrol route."""
    ordem = ["A", "B", "C", "D", "Base"]
    idx = ordem.index(posicao_atual)
    return ordem[(idx + 1) % len(ordem)]


def desenhar_sistema(sistema):
    """Draw the system state on the screen."""
    tela.fill(PRETO)

    # Draw locations
    for posicao, coordenadas in POSICOES.items():
        cor = VERDE if posicao == "Base" else AZUL
        pygame.draw.circle(tela, cor, coordenadas, 20)
        fonte = pygame.font.Font(None, 36)
        texto = fonte.render(posicao, True, BRANCO)
        tela.blit(texto, (coordenadas[0] - 10, coordenadas[1] - 50))

    # Draw obstacle
    pygame.draw.rect(
        tela,
        AMARELO,
        (
            LARGURA // 2 - OBSTACULO_LARGURA // 2,
            sistema["obstaculo_y"],
            OBSTACULO_LARGURA,
            OBSTACULO_ALTURA,
        ),
    )

    # Draw drone
    pygame.draw.circle(
        tela,
        VERMELHO,
        (int(sistema["posicao_drone"][0]), int(sistema["posicao_drone"][1])),
        10,
    )

    # Draw information
    fonte_info = pygame.font.Font(None, 24)
    info = f"Estado: {sistema['estado_drone']} | Posição Atual: {sistema['posicao_atual']} | Próximo Destino: {sistema['proximo_destino']}"
    texto_info = fonte_info.render(info, True, BRANCO)
    tela.blit(texto_info, (10, 10))


def simular_sistema():
    """Run the simulation."""
    sistema = representar_e_inicializar_o_sistema()
    clock = pygame.time.Clock()

    while True:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        sistema = regra_atualizar_estado_voo(sistema)
        desenhar_sistema(sistema)

        pygame.display.flip()
        clock.tick(30)  # 30 FPS for smooth animation


if __name__ == "__main__":
    simular_sistema()
