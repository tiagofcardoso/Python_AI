import torch
import pygame
import sys
import torch.nn as nn

n_in, n_h, n_out, batch_size = 10, 5, 1, 10
x = torch.randn(batch_size, n_in)
y = torch.tensor([[1.0], [0.0], [0.0], [1.0], [1.0],
                 [1.0], [0.0], [0.0], [1.0], [0.0]])

# definiçã da rede neural
model = nn.Sequential(nn.Linear(n_in, n_h),
                      nn.ReLU(),
                      nn.Linear(n_h, n_out),
                      nn.Sigmoid())

# Loss e otimizador
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# treinamento
for epoch in range(50):
    # Forward pass
    y_pred = model(x)
    # Compute Loss
    loss = criterion(y_pred, y)
    print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
    optimizer.zero_grad()
    # Backward pass
    loss.backward()
    optimizer.step()
   # Inicializa o Pygame
pygame.init()

# Configurações da janela
width, height = 800, 600
win = pygame.display.set_mode((width, height))
pygame.display.set_caption("Neural Network Output Visualization")

# Função para desenhar os outputs


def draw_output(outputs):
    win.fill((0, 0, 0))  # Limpa a tela
    font = pygame.font.SysFont(None, 55)
    for i, output in enumerate(outputs):
        text = font.render(
            f'Output {i+1}: {output.item():.2f}', True, (255, 255, 255))
        win.blit(text, (50, 50 + i * 60))
    pygame.display.update()


   # Loop principal do Pygame
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            pygame.quit()
            sys.exit()

    # Forward pass
    y_pred = model(x)
    # Compute Loss
    loss = criterion(y_pred, y)
    print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
    optimizer.zero_grad()
    # Backward pass
    loss.backward()
    optimizer.step()

    # Desenha os outputs
    draw_output(y_pred)

    # Delay para visualização
    pygame.time.delay(500)
