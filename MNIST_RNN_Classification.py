import torchvision
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

BATCH_SIZE = 64

transform = transforms.Compose([transforms.ToTensor()])

train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, BATCH_SIZE, shuffle=True, num_workers=2)

test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, BATCH_SIZE, shuffle=False, num_workers=2)

N_STEPS = 28
N_INPUTS = 28
N_NEURONS = 150
N_OUTPUT = 10
N_EPOCHS = 10


class ImageRNN(nn.Module):
    def __init__(self, batch_size, n_steps, n_inputs, n_neurons, n_outputs):
        super(ImageRNN, self).__init__()

        self.batch_size = batch_size
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.n_outputs = n_outputs

        self.basic_rnn = nn.RNN(self.n_inputs, self.n_neurons)

        self.FC_layer = nn.Linear(self.n_neurons, self.n_outputs)

    def initial_hidden(self,):
        return torch.zeros(1, self.batch_size, self.n_neurons)

    def forward(self, X):
        # transforms X to dimensions: n_steps X batch_size X n_inputs
        X = X.permute(1, 0, 2)

        self.batch_size = X.size(1)
        hidden = self.initial_hidden()

        lstm_out, hidden = self.basic_rnn(X, hidden)
        out = self.FC_layer(hidden)
        return out.view(-1, self.n_outputs)


def get_accuracy(logit, target, batch_size):
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects/batch_size
    return accuracy.item()


testing_code = False
if testing_code:
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    model = ImageRNN(BATCH_SIZE, N_STEPS, N_INPUTS, N_NEURONS, N_OUTPUT)

    # testing the data and output
    logits = model(images.view(-1, 28, 28))
    print(logits.size())
    print(logits[0:10])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ImageRNN(BATCH_SIZE, N_STEPS, N_INPUTS, N_NEURONS, N_OUTPUT)
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


for epoch in range(N_EPOCHS):
    training_running_loss = 0.0
    train_accuracy = 0.0
    model.train()

    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        model.hidden = model.initial_hidden()
        #print('data : ', data[0].shape)
        inputs, labels = data
        inputs = inputs.view(-1, 28, 28)
        #print('Input : ', inputs.shape)
        outputs = model(inputs)
        #print('Output: ', outputs.shape)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        training_running_loss += loss.detach().item()
        train_accuracy += get_accuracy(outputs, labels, BATCH_SIZE)

    model.eval()
    print('Epoch: %d | Loss: %.4f | Train Accuracy: %.2f' % (epoch, training_running_loss/i, train_accuracy/i))

test_accuracy = 0.0
for i, data in enumerate(test_loader, 0):
    inputs, labels = data
    inputs = inputs.view(-1, 28, 28)

    outputs = model(inputs)

    test_accuracy += get_accuracy(outputs, labels, BATCH_SIZE)

print("Test Accuracy: %.2f" %(test_accuracy/i))
