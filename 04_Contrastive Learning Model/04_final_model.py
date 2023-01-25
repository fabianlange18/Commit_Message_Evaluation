import wandb
import torch
import pandas as pd
import random
from torch.utils.data import DataLoader
import pytorch_lightning as L

import sys
sys.path.append('.')
from util.style_model import StyleModel

# Server GPU
# wandb.config = {
#   "batch_size": 512,
#   "learning_rate": 1e-3,
#   "max_length": 20,
#   "epochs": 15,
#   "num_workers": 48,
#   "precision": 16,
#   "accelerator": 'gpu',
#   "devices": 1,
#   "subset_size": 500000
# }

# Local MPS
wandb.config = {
  "batch_size": 64,
  "learning_rate": 1e-3,
  "max_length": 20,
  "epochs": 10,
  "num_workers": 8,
  "precision": 16,
  "accelerator": 'mps',
  "devices": 1,
  "subset_size": 1000
}

wandb.init(project="contrastive_model", entity="commit_message_evaluation", config = wandb.config)


### Load Data
# Build dataloader with saved files

# Pickle really slow

# with open('data/06a_Contrastive_Train_Pairs.pkl', 'rb') as f:
#     training_pairs = pickle.load(f)
# with open('data/06b_Contrastive_Validate_Pairs.pkl', 'rb') as f:
#    validate_pairs = pickle.load(f)
# with open('data/06c_Contrastive_Test_Pairs.pkl', 'rb') as f:
#     testing_pairs = pickle.load(f)

# Build Pairs by the function here

import sys
sys.path.append('.')
from util.contrastive_pairs import build_contrastive_pairs

# training_pairs = build_contrastive_pairs('data/04a_Train_Set.pkl', 369)
# testing_pairs = build_contrastive_pairs('data/04c_Test_Set.pkl', 647)

# Testing training pairs

train_data = pd.read_pickle('data/04a_Train_Set.pkl')
validate_data = pd.read_pickle('data/04b_Validate_Set.pkl')
test_data = pd.read_pickle('data/04c_Test_Set.pkl')

training_pairs = []
testing_pairs = []

for i, group in enumerate(train_data.groupby("author_email")):
    pair = []
    for i, message in enumerate(group[1]['message']):
        pair.append(message)
        if i % 2 == 1:
            pair.append(1 if random.choice([True, False]) else -1)
            training_pairs.append(pair)
            pair = []

for i, group in enumerate(test_data.groupby("author_email")):
    pair = []
    for i, message in enumerate(group[1]['message']):
        pair.append(message)
        if i % 2 == 1:
            pair.append(1 if random.choice([True, False]) else -1)
            testing_pairs.append(pair)
            pair = []

# Do this when taking a subset to ensure that shuffling happens before taking subsets
random.shuffle(training_pairs)
random.shuffle(testing_pairs)

train_dataloader = DataLoader(training_pairs[:wandb.config['subset_size']], wandb.config['batch_size'], shuffle=True, drop_last=True, num_workers=wandb.config['num_workers'])
test_dataloader = DataLoader(testing_pairs[:wandb.config['subset_size']], wandb.config['batch_size'], shuffle=False, drop_last=True, num_workers=wandb.config['num_workers'])

if __name__ == '__main__':
    model = StyleModel()
    wandb.watch(model)
    trainer = L.Trainer(
        accelerator=wandb.config['accelerator'],
        devices=wandb.config['devices'],
        max_epochs=wandb.config['epochs'],
        precision=wandb.config['precision']
    )
    trainer.fit(model, train_dataloader, test_dataloader)
    # torch.save(model.state_dict(), 'model/Style_Model.pt')