# Source for SBERT:
# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader

MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

from transformers import AutoTokenizer, AutoModel

train_data = pd.read_pickle('data/04a_Train_Set.pkl')
validate_data = pd.read_pickle('data/04b_Validate_Set.pkl')
test_data = pd.read_pickle('data/04c_Test_Set.pkl')

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


train_dataloader = DataLoader(training_pairs, batch_size)
test_dataloader = DataLoader(testing_pairs, batch_size)

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define SBert Model
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


class Model(nn.Module):
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

model = Model().to(device)
print(model)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X1, X2) in enumerate(dataloader):
        X1, X2 = X1.to(device), X2.to(device)

        # Compute prediction error
        X1_s = model(X1)
        X2_s = model(X2)

        # Introduce target (labels) to prepare for loss: simple case of only positive pairs -> always 1
        target = torch.full((X1_s.shape[0],), 1)

        # Compute Loss
        loss = loss_fn(X1_s, X2_s, target)

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
            # Bad practice: Initilize target again
            target = torch.full((X1_s.shape[0],), 1)
            test_loss += loss_fn(X1_s, X2_s, target).item()
    test_loss /= num_batches
    print(f"Avg loss: {test_loss:>8f} \n")


loss_fn = nn.CosineEmbeddingLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 10

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")