import torch
import wandb
import pytorch_lightning as L
from transformers import AutoTokenizer, AutoModel

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

class SBERT(L.LightningModule):
    def __init__(self, frozen = False):
        super().__init__()
        self.sbert = AutoModel.from_pretrained(MODEL)

        if frozen:
            for param in self.sbert.parameters():
                param.requires_grad = False

        # Tokenization
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)

    def tokenize_function(self, examples, max_length = 25):
        return self.tokenizer(examples, padding=True, truncation=True, return_tensors='pt', max_length=max_length)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, sentence): # sentence
        encoding = self.tokenize_function(sentence)
        encoding['attention_mask'] = encoding['attention_mask']
        encoding['input_ids'] = encoding['input_ids']
        encoding['token_type_ids'] = encoding['token_type_ids']

        embeddings = self.sbert(**encoding)
        # Perform pooling
        sentence_embeddings = self.mean_pooling(embeddings, encoding['attention_mask'])
        # Normalize embeddings
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings


class StyleModel(L.LightningModule):

    loss_fn = torch.nn.CosineEmbeddingLoss()

    def __init__(self) -> None:
        super().__init__()
        self.sbert_m_s = SBERT()
        self.sbert_m = SBERT(frozen = True)

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
        loss = self.loss_fn(X1_s, X2_s, target)
        
        # Log Loss to WandB
        wandb.log({"train_loss": loss})

        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        X1, X2, target = val_batch
        X1_s = self(X1)
        X2_s = self(X2)
        loss = self.loss_fn(X1_s, X2_s, target)
        # Log Loss to WandB
        wandb.log({"val_loss": loss})
        self.log('val_loss', loss, batch_size=wandb.config['batch_size'])