# Source for SBERT:
# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

import torch
import pickle
import random
import pandas as pd
import pytorch_lightning as L

from torch import nn
from torch.utils.data import DataLoader

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

from transformers import AutoTokenizer, AutoModel

import wandb

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

# Local
wandb.config = {
  "batch_size": 64,
  "learning_rate": 5e-3,
  "max_length": 20,
  "epochs": 10,
  "num_workers": 8,
  "precision": 16,
  "accelerator": 'mps',
  "devices": 1,
  "subset_size": 1000
}

wandb.init(project="contrastive_model", entity="commit_message_evaluation", config = wandb.config)

######### Helper functions #########
# Tokenization
tokenizer = AutoTokenizer.from_pretrained(MODEL)

def tokenize_function(examples):
    return tokenizer(examples, padding=True, truncation=True, return_tensors='pt', max_length=wandb.config['max_length'])

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

######### End Helper functions #########


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


# training_pairs_encoding = [[tokenize_function(sentence1), tokenize_function(sentence2), target] for sentence1, sentence2, target in training_pairs]
# testing_pairs_encoding = [[tokenize_function(sentence1), tokenize_function(sentence2), target] for sentence1, sentence2, target in testing_pairs]

# Do this when taking a subset to ensure that shuffling happens before taking subsets
random.shuffle(training_pairs)
random.shuffle(testing_pairs)

train_dataloader = DataLoader(training_pairs[:wandb.config['subset_size']], wandb.config['batch_size'], shuffle=True, drop_last=True, num_workers=wandb.config['num_workers'])
test_dataloader = DataLoader(testing_pairs[:wandb.config['subset_size']], wandb.config['batch_size'], shuffle=False, drop_last=True, num_workers=wandb.config['num_workers'])

# Define SBert Model
class SBERT(L.LightningModule):
    def __init__(self, frozen = False):
        super().__init__()
        self.sbert = AutoModel.from_pretrained(MODEL)
        if frozen:
            for param in self.sbert.parameters():
                param.requires_grad = False

    def forward(self, sentence): # sentence
        encoding = tokenize_function(sentence)
        encoding['attention_mask'] = encoding['attention_mask'].to(self.device)
        encoding['input_ids'] = encoding['input_ids'].to(self.device)
        encoding['token_type_ids'] = encoding['token_type_ids'].to(self.device)

        embeddings = self.sbert(**encoding)
        # Perform pooling
        sentence_embeddings = mean_pooling(embeddings, encoding['attention_mask'])
        # Normalize embeddings
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings

loss_fn = nn.CosineEmbeddingLoss()

class StyleModel(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.sbert_m_s = SBERT()
        self.sbert_m = SBERT(frozen=True)

    def forward(self, message):
        embeddings_m_s = self.sbert_m_s(message)
        embeddings_m = self.sbert_m(message)
        # maybe use a feed-forward layer here instead
        embeddings_s = embeddings_m_s - embeddings_m
        return embeddings_s

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.sbert_m_s.parameters(), lr=wandb.config['learning_rate'])
        return optimizer

    def training_step(self, train_batch, batch_idx):
        X1, X2, target = train_batch

        # Compute prediction error
        X1_s = self(X1)
        X2_s = self(X2)

        # Compute Loss
        loss = loss_fn(X1_s, X2_s, target)
        
        # Log Loss to WandB
        wandb.log({"train_loss": loss})

        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        X1, X2, target = val_batch
        X1_s = self(X1)
        X2_s = self(X2)
        loss = loss_fn(X1_s, X2_s, target)
        # Log Loss to WandB
        wandb.log({"val_loss": loss})
        self.log('val_loss', loss, batch_size=wandb.config['batch_size'])


if __name__ == '__main__':
    lightning_model = StyleModel()
    wandb.watch(lightning_model)
    trainer = L.Trainer(
        accelerator=wandb.config['accelerator'],
        devices=wandb.config['devices'],
        max_epochs=wandb.config['epochs'],
        precision=wandb.config['precision'])
    trainer.fit(lightning_model, train_dataloader, test_dataloader)
    torch.save(lightning_model.state_dict(), 'model/Style_Model.pt')