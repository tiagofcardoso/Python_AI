import pygame
import sys
import math

# Inicializar Pygame
pygame.init()

# Configurações da tela
LARGURA, ALTURA = 800, 600
tela = pygame.display.set_mode((LARGURA, ALTURA))
pygame.display.set_caption("Simulação de Drone - Ronda com Obstáculo")

# Cores
PRETO = (0, 0, 0)
BRANCO = (255, 255, 255)
VERMELHO = (255, 0, 0)
VERDE = (0, 255, 0)
AZUL = (0, 0, 255)
AMARELO = (255, 255, 0)

# Posições dos locais
POSICOES = {
    'Base': (400, 550),
    'A': (200, 200),
    'B': (600, 200),
    'C': (200, 400),
    'D': (600, 400)
}

# Configurações do obstáculo
OBSTACULO_LARGURA = 20
OBSTACULO_ALTURA = 200
OBSTACULO_VELOCIDADE = 2

def representar_e_inicializar_o_sistema():
    return {
        'estado_drone': 'Descolagem',
        'posicao_atual': 'Base',
        'posicao_drone': POSICOES['Base'],
        'destino': POSICOES['A'],
        'proximo_destino': 'A',
        'movimento_completo': False,
        'obstaculo_y': (ALTURA - OBSTACULO_ALTURA) // 2,
        'obstaculo_direcao': 1,  # 1 para baixo, -1 para cima
        'rota_alternativa': None,
        'evitando_obstaculo': False
    }

def regra_atualizar_estado_voo(sistema):
    sistema = regra_mover_drone(sistema)

    if sistema['movimento_completo'] and sistema['estado_drone'] == 'Descolagem':
        sistema['estado_drone'] = 'Em Rota'
    if sistema['movimento_completo'] and sistema['estado_drone'] == 'Em Rota' and not sistema['evitando_obstaculo']:
        sistema['posicao_atual'] = sistema['proximo_destino']
        proxima_posicao = ação_obter_proxima_posicao(sistema['posicao_atual'])
        sistema['proximo_destino'] = proxima_posicao
        sistema['destino'] = POSICOES[proxima_posicao]
        sistema['movimento_completo'] = False
        sistema['rota_alternativa'] = None
    
    # Atualizar posição do obstáculo
    sistema['obstaculo_y'] += OBSTACULO_VELOCIDADE * sistema['obstaculo_direcao']
    if sistema['obstaculo_y'] <= 0 or sistema['obstaculo_y'] + OBSTACULO_ALTURA >= ALTURA:
        sistema['obstaculo_direcao'] *= -1
    
    return sistema

def regra_mover_drone(sistema):
    velocidade = 5
    x1, y1 = sistema['posicao_drone']
    x2, y2 = sistema['destino']
    
    dx = x2 - x1
    dy = y2 - y1
    distancia = math.sqrt(dx**2 + dy**2)
    
    if distancia >= velocidade:
        fator = velocidade / distancia
        novo_x = x1 + dx * fator
        novo_y = y1 + dy * fator

        sistema['posicao_drone'] = (novo_x, novo_y)

    if distancia < velocidade:
        sistema['posicao_drone'] = sistema['destino']
        sistema['movimento_completo'] = True
        sistema['evitando_obstaculo'] = False
    
    return sistema

def ação_obter_proxima_posicao(posicao_atual):
    ordem = ['A', 'B', 'C', 'D', 'Base']
    idx = ordem.index(posicao_atual)
    return ordem[(idx + 1) % len(ordem)]

def desenhar_sistema(sistema):
    tela.fill(PRETO)
    
    # Desenhar locais
    for posicao, coordenadas in POSICOES.items():
        cor = VERDE if posicao == 'Base' else AZUL
        pygame.draw.circle(tela, cor, coordenadas, 20)
        fonte = pygame.font.Font(None, 36)
        texto = fonte.render(posicao, True, BRANCO)
        tela.blit(texto, (coordenadas[0] - 10, coordenadas[1] - 50))
    
    # Desenhar obstáculo
    pygame.draw.rect(tela, AMARELO, (LARGURA // 2 - OBSTACULO_LARGURA // 2, sistema['obstaculo_y'], OBSTACULO_LARGURA, OBSTACULO_ALTURA))
    
    # Desenhar drone
    pygame.draw.circle(tela, VERMELHO, (int(sistema['posicao_drone'][0]), int(sistema['posicao_drone'][1])), 10)
    
    # Desenhar informações
    fonte_info = pygame.font.Font(None, 24)
    info = f"Estado: {sistema['estado_drone']} | Posição Atual: {sistema['posicao_atual']} | Próximo Destino: {sistema['proximo_destino']}"
    texto_info = fonte_info.render(info, True, BRANCO)
    tela.blit(texto_info, (10, 10))

def simular_sistema():
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
        clock.tick(30)  # 30 FPS para uma animação suave

if __name__ == '__main__':
    simular_sistema()
