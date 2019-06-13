# Short practice using PyTorch to build a basic neural net with Sigmoid activation

import torch
import numpy as np

def activation(x):
    """ Sigmoid activation function

        Arguments
        ---------
        x: torch.Tensor
    """
    return 1/(1+torch.exp(-x))

torch.manual_seed(7)
features = torch.randn((1, 3))

# Define input layers, weights & bias
n_input = features.shape[1]
n_hidden = 2
n_output = 1
W1 = torch.randn(n_input, n_hidden)
W2 = torch.randn(n_hidden, n_output)
B1 = torch.randn((1, n_hidden))
B2 = torch.randn((1, n_output))

h = torch.mm(features, W1)+B1
y = activation(torch.mm(h, W2)+B2)
print(y)

# Practice moving from Numpy to Torch and back
a = np.random.rand(4,3)
# print(a)
b = torch.from_numpy(a)
# print(b)
c = b.numpy()
# print(c)