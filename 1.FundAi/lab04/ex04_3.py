import torch

# Broadcasting
x = torch.empty(5, 1, 4, 1)
y = torch.empty(3, 1, 1)
print("Broadcasting")
print((x + y).size())

# Redimensionamento
x = torch.arange(8)
x_reshaped = x.view(2, 4)
print("Redimensionamento")
print(x_reshaped)
print(x_reshaped.size())