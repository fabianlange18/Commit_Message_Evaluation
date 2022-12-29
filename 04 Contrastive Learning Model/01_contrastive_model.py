# Source for SBERT:
# https://huggingface.co/NbAiLab/nb-sbert-base

import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader

from pytorch_metric_learning import losses

from sentence_transformers import SentenceTransformer

train_data = pd.read_pickle('data/04a_Train_Set.pkl')
validate_data = pd.read_pickle('data/04b_Validate_Set.pkl')
test_data = pd.read_pickle('data/04c_Test_Set.pkl')

batch_size = 64

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

train_dataloader = DataLoader(training_pairs, batch_size)
test_dataloader = DataLoader(testing_pairs, batch_size)

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define SBert Model
class SBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.sbert = SentenceTransformer('NbAiLab/nb-bert-base')

    def forward(self, sentence):
        embedding = self.sbert.encode(sentence)
        return embedding


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sbert_m_s = SBERT()
        self.sbert_m = SBERT()

    def forward(self, message):
        embedding_m_s = self.sbert_m_s(message)
        with torch.no_grad():
            embedding_m = self.sbert_m(message)
        # maybe use a feed-forward layer here instead
        embedding_s = embedding_m_s - embedding_m
        return embedding_s

model = Model().to(device)
print(model)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X1, X2) in enumerate(dataloader):
        # X1, X2 = X1.to(device), X2.to(device)

        # Compute prediction error
        X1_s = model(X1)
        X2_s = model(X2)

        # Prepare for loss
        embeddings = torch.cat((torch.from_numpy(X1_s).requires_grad_(), torch.from_numpy(X2_s).requires_grad_()))
        # Is the requires grad necessary?
        # Or should this be added to the model parameters in SBERT and is not done because of the use of Sentence Transformer?
        indices = torch.arange(0, torch.from_numpy(X1_s).size(0), device=torch.from_numpy(X2_s).device)
        labels = torch.cat((indices, indices))

        # Compute Loss
        loss = loss_fn(embeddings, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 5 == 0:
            loss, current = loss.item(), batch * len(X1)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X1, X2 in dataloader:
            # X1, X2 = X1.to(device), X2.to(device)
            X1_s = model(X1)
            X2_s = model(X2)
            test_loss += loss_fn(X1_s, X2_s).item()
    test_loss /= num_batches
    print(f"Avg loss: {test_loss:>8f} \n")


# loss_fn = nn.CosineEmbeddingLoss(reduction='none')
loss_fn = losses.ContrastiveLoss(pos_margin=0.2, neg_margin=0.8)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 10

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")