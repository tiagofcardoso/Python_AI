import torch
x = torch.rand(5, 3)
print(x)

# Tensor de zeros de 5x3 tipo long
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# Tensoer diretamente de dados
x = torch.tensor([5.5, 3])
print(x)
