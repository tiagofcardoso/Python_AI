import pygame
import sys
import math

# Inicializar Pygame
pygame.init()

# Configurações da tela
LARGURA, ALTURA = 800, 600 #Largura e Altura do Ecrã
tela = pygame.display.set_mode((LARGURA, ALTURA))
pygame.display.set_caption("Simulação de Drone - Ronda com Obstáculo e Detecção de Colisão")

# Cores definidas como tuplos RGB
PRETO = (0, 0, 0)
BRANCO = (255, 255, 255)
VERMELHO = (255, 0, 0)
VERDE = (0, 255, 0)
AZUL = (0, 0, 255)
AMARELO = (255, 255, 0)

# Posições dos locais, predefinidas como os locais a visitar/navegar
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
RAIO_DRONE = 10

def verificar_colisao(posicao_drone, obstaculo_y):
    """
    Verifica se há colisão entre o drone e o obstáculo em movimento
    Cria rectângulos para tanto o drone como o obstáculo
    Utiliza o sistema de deteção de colisão do Pygame
    """
    drone_x, drone_y = posicao_drone
    obstaculo_x = LARGURA // 2 - OBSTACULO_LARGURA // 2
    
    drone_rect = pygame.Rect(
        drone_x - RAIO_DRONE,
        drone_y - RAIO_DRONE,
        RAIO_DRONE * 2,
        RAIO_DRONE * 2
    )
    
    obstaculo_rect = pygame.Rect(
        obstaculo_x,
        obstaculo_y,
        OBSTACULO_LARGURA,
        OBSTACULO_ALTURA
    )
    
    return drone_rect.colliderect(obstaculo_rect)

def calcular_rota_alternativa(sistema):
    """
    Calcula uma rota alternativa quando detectada colisão iminente
    Se o drone está à esquerda do obstáculo, redireciona à volta pelo lado esquerdo
    Se o drone está à direita do obstáculo, redireciona à volta pelo lado direito 
    """
    drone_x, drone_y = sistema['posicao_drone']
    obstaculo_x = LARGURA // 2
    destino_x, destino_y = sistema['destino']
    
    # Decide se deve desviar pela esquerda ou direita do obstáculo
    if drone_x < obstaculo_x:
        # Se o drone está à esquerda do obstáculo
        ponto_desvio = (
            obstaculo_x - OBSTACULO_LARGURA - 10,
            (drone_y + destino_y) // 2
        )
    else:
        # Se o drone está à direita do obstáculo
        ponto_desvio = (
            obstaculo_x + OBSTACULO_LARGURA + 10,
            (drone_y + destino_y) // 2
        )
    
    return ponto_desvio

def prever_colisao(sistema):
    """
    Prevê se haverá colisão no caminho atual do drone
    Deteta a posição atual (x1, y1) e o destino (x2, y2) do sistema
    
    Utiliza interpolação linear para verificar 10 pontos ao longo do caminho do drone
        - t vai de 0 a 0.9 em passos de 0.1
        - quando t = 0, verifica a posição inicial
        - quando t = 0.5, verifica ponto intermédio
        - quando t = 0.9, verifica perto do fim
    
    Para cada ponto:
        - x = x1 + (x2 - x1) * t : cálculo das coordenadas x
        - y = y1 + (y2 - y1) * t : cálculo das coordenadas y
        - estas fórmulas criam espaçamentos distribuídos uniformemente ao longo da reta entre início e destino
    
    Em cada ponto, verifica se há colisão com recurso à função verificar_colisao
        - se algum ponto resultar em colisão, devolve True
        - se não verificar colisões, devolve False
    """
    x1, y1 = sistema['posicao_drone']
    x2, y2 = sistema['destino']
    
    for i in range(10):
        t = i / 10
        x = x1 + (x2 - x1) * t
        y = y1 + (y2 - y1) * t
        
        if verificar_colisao((x, y), sistema['obstaculo_y']):
            return True
    
    return False

def representar_e_inicializar_o_sistema():
    """
    Retorna dicionário do estado inicial do sistema com:
    - Estado do Drone ('Descolagem')
    - Posição Atual (inicia em 'Base')
    - Destino
    - Posição de Obstáculo e Direção
    - Vários alertas para estado de movimento
    """
    return {
        'estado_drone': 'Descolagem',
        'posicao_atual': 'Base',
        'posicao_drone': POSICOES['Base'],
        'destino': POSICOES['A'],
        'destino_original': None,
        'proximo_destino': 'A',
        'movimento_completo': False,
        'obstaculo_y': (ALTURA - OBSTACULO_ALTURA) // 2,
        'obstaculo_direcao': 1,
        'rota_alternativa': None,
        'evitando_obstaculo': False
    }

def regra_atualizar_estado_voo(sistema):
    # Verifica colisão iminente
    """
    Lida com previsão de colisão e de avoid
    Atualiza o movimento do drone
    Atualiza o movimento do obstáculo
    Gere as transições de estado
    """
    if not sistema['evitando_obstaculo'] and sistema['estado_drone'] != 'Colisão':
        if prever_colisao(sistema):
            sistema['evitando_obstaculo'] = True
            sistema['rota_alternativa'] = calcular_rota_alternativa(sistema)
            sistema['destino_original'] = sistema['destino']
            sistema['destino'] = sistema['rota_alternativa']
    
    # Se está evitando obstáculo e chegou ao ponto de desvio
    if sistema['evitando_obstaculo'] and sistema['movimento_completo']:
        sistema['destino'] = sistema['destino_original']
        sistema['rota_alternativa'] = None
        sistema['evitando_obstaculo'] = False
        sistema['movimento_completo'] = False
    
    # Atualiza movimento do drone
    sistema = regra_mover_drone(sistema)
    
    # Verifica se houve colisão real
    if verificar_colisao(sistema['posicao_drone'], sistema['obstaculo_y']):
        sistema['estado_drone'] = 'Colisão'
        return sistema
    
    # Atualizar estado normal do voo
    if sistema['movimento_completo'] and sistema['estado_drone'] == 'Descolagem':
        sistema['estado_drone'] = 'Em Rota'
    if (sistema['movimento_completo'] and 
        sistema['estado_drone'] == 'Em Rota' and 
        not sistema['evitando_obstaculo']):
        sistema['posicao_atual'] = sistema['proximo_destino']
        proxima_posicao = ação_obter_proxima_posicao(sistema['posicao_atual'])
        sistema['proximo_destino'] = proxima_posicao
        sistema['destino'] = POSICOES[proxima_posicao]
        sistema['movimento_completo'] = False
    
    # Atualizar posição do obstáculo
    sistema['obstaculo_y'] += OBSTACULO_VELOCIDADE * sistema['obstaculo_direcao']
    if (sistema['obstaculo_y'] <= 0 or 
        sistema['obstaculo_y'] + OBSTACULO_ALTURA >= ALTURA):
        sistema['obstaculo_direcao'] *= -1
    
    return sistema

def regra_mover_drone(sistema):
    """
    Calcula o movimento passo a passo rumo ao destino
    Utiliza cálculos de velocidade e distância
    Atualiza gradualmente a posição do drone
    """
    if sistema['estado_drone'] == 'Colisão':
        return sistema
        
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
    else:
        sistema['posicao_drone'] = sistema['destino']
        sistema['movimento_completo'] = True
    
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
    pygame.draw.rect(tela, AMARELO, 
                    (LARGURA // 2 - OBSTACULO_LARGURA // 2,
                     sistema['obstaculo_y'],
                     OBSTACULO_LARGURA,
                     OBSTACULO_ALTURA))
    
    # Desenhar rota alternativa se estiver evitando obstáculo
    if sistema['evitando_obstaculo']:
        pygame.draw.line(tela, AMARELO,
                        sistema['posicao_drone'],
                        sistema['rota_alternativa'], 2)
        pygame.draw.line(tela, AMARELO,
                        sistema['rota_alternativa'],
                        sistema['destino_original'], 2)
    
    # Desenhar drone
    pygame.draw.circle(tela, VERMELHO,
                      (int(sistema['posicao_drone'][0]),
                       int(sistema['posicao_drone'][1])),
                      RAIO_DRONE)
    
    # Desenhar informações
    fonte_info = pygame.font.Font(None, 24)
    info = f"Estado: {sistema['estado_drone']} | Posição Atual: {sistema['posicao_atual']} | Próximo Destino: {sistema['proximo_destino']}"
    texto_info = fonte_info.render(info, True, BRANCO)
    tela.blit(texto_info, (10, 10))
    
    # Se houver colisão, desenhar alerta
    if sistema['estado_drone'] == 'Colisão':
        fonte = pygame.font.Font(None, 72)
        texto = fonte.render("COLISÃO!", True, VERMELHO)
        tela.blit(texto, (LARGURA//2 - 100, ALTURA//2))

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
        clock.tick(30) # 30 FPS para uma animação suave

if __name__ == '__main__':
    simular_sistema()