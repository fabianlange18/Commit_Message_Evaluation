# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

import torch
from torch import nn
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np

train_data = pd.read_pickle('data/05a_Authors_Train_Set.pkl')
test_data = pd.read_pickle('data/05c_Authors_Test_Set.pkl')

batch_size = 64

train_featureset = np.load('./data/05a_Train_Set_Features.npy', allow_pickle=True)
test_featureset = np.load('./data/05c_Test_Set_Features.npy', allow_pickle=True)

train_data_tensors = torch.tensor(train_featureset, dtype=torch.float32)
test_data_tensors = torch.tensor(test_featureset, dtype=torch.float32)

train_tensors_labeled = [[tensor, int(train_data['label'][i])] for i, tensor in enumerate(train_data_tensors)]
test_tensors_labeled = [[tensor, int(test_data['label'][i])] for i, tensor in enumerate(test_data_tensors)]

train_dataloader = DataLoader(train_tensors_labeled, batch_size)
test_dataloader = DataLoader(test_tensors_labeled, batch_size)


# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(25, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 42),
            nn.ReLU()
        )
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        prediction = self.linear_relu_stack(x)
        class_prediction = self.softmax(prediction)
        return class_prediction

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")



def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")