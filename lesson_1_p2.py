# Using a multi-layer network to classify images
import torch
import helper
import numpy as np
import helper
import matplotlib as plt
from torchvision import datasets, transforms
from torch import nn
import torch.nn.functional as F

def activation(x):
    """ Sigmoid activation function """
    return 1/(1+torch.exp(-x))

def softmax(x):
    """ Softmax activation function """
    b = torch.sum(torch.exp(x), dim=1)
    d = b.view(x.shape[0], 1)
    return torch.exp(x)/d

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = dataiter.next()


images_flat = images.view(64, 784)

# -----------------------------------------------------

# Constructing the neural net (manually)
n_input = images_flat.shape[1]
n_hidden = 256
n_output = 10
W1 = torch.randn(n_input, n_hidden)
W2 = torch.randn(n_hidden, n_output)
B1 = torch.randn((64, n_hidden))
B2 = torch.randn((64, n_output))

h = activation(torch.mm(images_flat, W1)+B1)
out = torch.mm(h, W2) + B2

probabilities = softmax(out)

# -----------------------------------------------------

# Constructing the neural net (using nn)
class Network_1(nn.Module):
    def __init__(self):
        super().__init__()

        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(784, 256)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(256, 10)

        # Define sigmoid activation and softmax output
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = F.sigmoid(self.hidden(x))
        x = F.softmax(self.output(x), dim=1)

        return x

# -----------------------------------------------------

# Constructing neural network with ReLu activation

# class Network(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.hidden_1 = nn.Linear(784, 128)
#         self.hidden_2 = nn.Linear(128, 64)
#         self.output = nn.Linear(64, 10)
#
#         self.softmax = nn.Softmax(dim=1)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x = F.relu(self.hidden_1(x))
#         x = F.relu(self.hidden_2(x))
#         x = F.softmax(self.output(x))
#
#         return x
#
# model = Network()
# print(model)
#
# print(model.hidden_1.weight.shape)

# -----------------------------------------------------

# Constructing neural network with nn.Sequential

input_size = 784
hidden_sizes = [128, 64]
output_size = 10

# Build a feed-forward network
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.Softmax(dim=1))
print(model)
images, labels = next(iter(trainloader))
images.resize_(images.shape[0], 1, 784)
ps = model.forward(images[0,:])
helper.view_classify(images[0].view(1, 28, 28), ps)