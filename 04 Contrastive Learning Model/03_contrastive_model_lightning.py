# Source for SBERT:
# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader

import pytorch_lightning as L

MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

from transformers import AutoTokenizer, AutoModel

train_data = pd.read_pickle('data/04a_Train_Set.pkl')
validate_data = pd.read_pickle('data/04b_Validate_Set.pkl')
test_data = pd.read_pickle('data/04c_Test_Set.pkl')


import wandb
wandb.init(project="contrastive_model", entity="commit_message_evaluation")

batch_size = 64


######### Helper functions #########

# Tokenization
tokenizer = AutoTokenizer.from_pretrained(MODEL)

def tokenize_function(examples):
    return tokenizer(examples, padding=True, truncation=True, return_tensors='pt')

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

######### End Helper functions #########


# Build dataloader with pairs
training_pairs = []
testing_pairs = []

for i, group in enumerate(train_data.groupby("author_email")):
    pair = []
    for i, message in enumerate(group[1]['message']):
        pair.append(message)
        if i % 2 == 1:
            training_pairs.append(pair)
            pair = []

for i, group in enumerate(test_data.groupby("author_email")):
    pair = []
    for i, message in enumerate(group[1]['message']):
        pair.append(message)
        if i % 2 == 1:
            testing_pairs.append(pair)
            pair = []

train_dataloader = DataLoader(training_pairs, batch_size, drop_last=True)
test_dataloader = DataLoader(testing_pairs, batch_size, drop_last=True)

# Define SBert Model
# This stays a torch module intentionally to only call it by the bigger pytorch module
class SBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.sbert = AutoModel.from_pretrained(MODEL)

    def forward(self, sentence):
        encoding = tokenize_function(sentence)
        embeddings = self.sbert(**encoding)
        # Perform pooling
        sentence_embeddings = mean_pooling(embeddings, encoding['attention_mask'])
        # Normalize embeddings
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings


loss_fn = nn.CosineEmbeddingLoss()

class LightningModel(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.sbert_m_s = SBERT()
        self.sbert_m = SBERT()

    def forward(self, message):
        embeddings_m_s = self.sbert_m_s(message)
        with torch.no_grad():
            embeddings_m = self.sbert_m(message)
        # maybe use a feed-forward layer here instead
        embeddings_s = embeddings_m_s - embeddings_m
        return embeddings_s

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        X1, X2 = train_batch

        # Compute prediction error
        X1_s = self(X1)
        X2_s = self(X2)

        # Introduce target (labels) to prepare for loss: simple case of only positive pairs -> always 1
        target = torch.full((X1_s.shape[0],), 1)

        # Compute Loss
        loss = loss_fn(X1_s, X2_s, target)
        
        # Log Loss to WandB
        wandb.log({"train_loss": loss})

        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        X1, X2 = val_batch
        X1_s = self(X1)
        X2_s = self(X2)
        target = torch.full((X1_s.shape[0],), 1)
        loss = loss_fn(X1_s, X2_s, target)
        # Log Loss to WandB
        wandb.log({"val_loss": loss})
        self.log('val_loss', loss, batch_size=batch_size)

    # If uncommented, this causes an error, if not, everything runs.
    # Is this required?
    # def backward(self, trainer, loss, optimizer, optimizer_idx):
    #    loss.backward()


lightning_model = LightningModel()
wandb.watch(lightning_model)
trainer = L.Trainer()
trainer.fit(lightning_model, train_dataloader, test_dataloader)