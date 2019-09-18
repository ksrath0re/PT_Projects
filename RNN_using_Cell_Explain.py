import torch
import torch.nn as nn
import numpy

rnn = nn.RNNCell(3, 5)  # n_input X n_neurons

X_batch = torch.tensor([[[0, 1, 2], [3, 4, 5],
                         [6, 7, 8], [9, 0, 1]],
                        [[9, 8, 7], [0, 0, 0],
                         [6, 5, 4], [3, 2, 1]]], dtype=torch.float)  # X0 and X1

print(X_batch.shape)
hx = torch.randn(4, 5)
output = []

# for each time step
for i in range(2):
    hx = rnn(X_batch[i], hx)
    output.append(hx)

print('output at different time stamp : \n', output)
