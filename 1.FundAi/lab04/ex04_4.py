import torch

a = torch.ones(5)
b = a.numpy()


# Detach e clonagem
c = a.clone().detach()
print("Tensor pytorch \n", a)
print("Numpy array \n", b)
print("Clone e detach \n", c)