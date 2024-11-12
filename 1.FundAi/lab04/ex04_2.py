import torch

# Criando dois tensores 
a = torch.rand(5, 3)
b = torch.rand(5, 3)

# Soma
print("Soma")
print(torch.add(a, b))

# Subtração
print("Subtração")
print(torch.sub(a, b))

# Multiplicação
mat1 = torch.randn(2, 3)
mat2 = torch.randn(3, 3)
print("Multiplicação")
print(torch.mm(mat1, mat2))
