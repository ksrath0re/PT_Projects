import torch
import torch.nn as nn
import numpy


class BasicSimpleRNNCell(nn.Module):
    def __init__(self, batch_size, n_inputs, n_neurons):
        super(BasicSimpleRNNCell, self).__init__()

        self.rnn = nn.RNNCell(n_inputs, n_neurons)
        self.hx = torch.randn(batch_size, n_neurons)

    def forward(self, X):
        output = []

        # for each timestamp
        for i in range(2):
            self.hx = self.rnn(X[i], self.hx)
            output.append(self.hx)

        return output, self.hx


FIXED_BATCH_SIZE = 4
N_INPUT = 3
N_NEURONS = 5

X_batch = torch.tensor([[[0, 1, 2], [3, 4, 5],
                         [6, 7, 8], [9, 0, 1]],
                        [[9, 8, 7], [0, 0, 0],
                         [6, 5, 4], [3, 2, 1]]
                        ], dtype=torch.float)  # X0 and X1

model1 = BasicSimpleRNNCell(FIXED_BATCH_SIZE, N_INPUT, N_NEURONS)
output_value, states_value = model1(X_batch)
print('All Output for all timesteps \n', output_value)
print('All states value for final timestep \n', states_value)
