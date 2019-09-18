import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy


class SingleRNN(nn.Module):
    def __init__(self, n_inputs, n_neurons):
        super(SingleRNN, self).__init__()

        self.Wx = torch.randn(n_inputs, n_neurons)
        self.Wy = torch.randn(n_neurons, n_neurons)

        self.b = torch.zeros(1, n_neurons)

    def forward(self, x0, x1):
        y0 = torch.tanh(torch.mm(x0, self.Wx) + self.b)
        y1 = torch.tanh(torch.mm(y0, self.Wy) + torch.mm(x1, self.Wx) + self.b)

        return y0, y1


N_INPUT = 3  # Number of features in Input
N_NEURONS = 7

x0_batch = torch.tensor([[0, 1, 2], [3, 4, 5],
                         [6, 7, 8], [9, 0, 1],
                         [6, 9, 8], [0, 0, 1]], dtype=torch.float)  # 4 x 3

x1_batch = torch.tensor([[9, 8, 7], [0, 0, 0],
                         [6, 5, 4], [3, 2, 1],
                         [6, 4, 3], [3, 1, 3]], dtype=torch.float)

model = SingleRNN(N_INPUT, N_NEURONS)
y0_val, y1_val = model(x0_batch, x1_batch)
print(y0_val)
print(y1_val)
