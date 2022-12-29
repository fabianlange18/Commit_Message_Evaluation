# This notebook is a version of the original one but containing only a training loop 
# that can be called be W&B Sweep Agents for Hyperparameter Search.


# Tutorial for Simple MLP
# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
# Tutorial for W&B Sweeps
# https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb

import torch
from torch import nn
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np

import wandb

# Config for sweep
sweep_config = {
    'method': 'random',
    'metric': {
        'name': 'loss',
        'goal': 'minimize' 
    },
    'parameters': {
        'learning_rate': {
            # a flat distribution between 0 and 0.1
            'distribution': 'uniform',
            'min': 0,
            'max': 0.1
        },
        'epochs': {
            'values': [2, 3, 4]
        },
        'batch_size': {
            # integers between 32 and 256
            # with evenly-distributed logarithms 
            'distribution': 'q_log_uniform_values',
            'q': 8,
            'min': 32,
            'max': 256,
        },
        'number_hidden_layers': {
            'distribution': 'q_uniform',
            'q': 5,
            'min': 0,
            'max': 20
        },
        'hidden_layer_size': {
            'distribution': 'q_uniform',
            'q': 5,
            'min': 10,
            'max': 20
        }
    }
}

# Wind up Sweep Controller
sweep_id = wandb.sweep(sweep_config, project="authors_clustering")

def load_data(batch_size):
    train_data = pd.read_pickle('../data/05a_Authors_Train_Set.pkl')
    test_data = pd.read_pickle('../data/05c_Authors_Test_Set.pkl')

    train_featureset = np.load('../data/05a_Train_Set_Features.npy', allow_pickle=True)
    test_featureset = np.load('../data/05c_Test_Set_Features.npy', allow_pickle=True)

    train_data_tensors = torch.tensor(train_featureset, dtype=torch.float32)
    test_data_tensors = torch.tensor(test_featureset, dtype=torch.float32)

    train_tensors_labeled = [[tensor, int(train_data['label'][i])] for i, tensor in enumerate(train_data_tensors)]
    test_tensors_labeled = [[tensor, int(test_data['label'][i])] for i, tensor in enumerate(test_data_tensors)]

    train_dataloader = DataLoader(train_tensors_labeled, batch_size)
    test_dataloader = DataLoader(test_tensors_labeled, batch_size)

    return train_dataloader, test_dataloader


# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Input Layer
        self.layers = [
            nn.Linear(25, config['hidden_layer_size']),
            nn.ReLU(),
            nn.BatchNorm1d(config['hidden_layer_size'])
        ]

        # Hidden Layers
        for _ in range(config['number_hidden_layers']):
            self.layers += [
                nn.Linear(config['hidden_layer_size'], config['hidden_layer_size']),
                nn.ReLU(),
                nn.BatchNorm1d(config['hidden_layer_size'])
            ]
        
        # Output Layer
        self.layers += [nn.Linear(config['hidden_layer_size'], 42), nn.Softmax(1)]

        # Concat
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)

def build_model(config):
    model = NeuralNetwork(config).to(device)
    wandb.watch(model)
    print(model)
    return model

def train(config=None):
    with wandb.init(config=config):
        config = wandb.config

        dataloader, _ = load_data(config['batch_size'])

        model = build_model(config)

        loss_fn = nn.CrossEntropyLoss()
        
        optimizer = torch.optim.SGD(model.parameters(), lr=wandb.config["learning_rate"])

        model.train()

        for epoch in range(config.epochs):
            avg_loss = 0
            cumu_loss = 0
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()

                # Compute prediction error
                pred = model(X)
                loss = loss_fn(pred, y)
                cumu_loss += loss.item()

                # Backpropagation
                loss.backward()
                optimizer.step()

                wandb.log( { "batch_loss": loss.item() } )

                if batch % 100 == 0:
                    loss = loss.item()
                    print(f"loss: {loss:>7f}  batch number: {batch:>3}")
            
            avg_loss = cumu_loss / len(dataloader.dataset)

            wandb.log({"loss": avg_loss, "epoch": epoch})

wandb.agent(sweep_id, train, count=5)