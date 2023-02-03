import random
import wandb
import torch
from torch import nn
import pytorch_lightning as L
from datasets.dataset_dict import DatasetDict
from transformers import AutoTokenizer, AutoModel, BertConfig

import logging
logging.basicConfig(level=logging.INFO)

# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
sys.path.append('.')
from util.contrastive_pairs import build_contrastive_pairs_data_dict
from util.tokenization import TokenizationWrapper, mean_pooling


# Server GPU
wandb.config = {
  "model_name": 'LargerSubset_TestData_StyleModel.pt',
  "batch_size": 256,
  "learning_rate": 1e-4,
  "max_length": 25,
  "epochs": 20,
  "precision": 16,
  "accelerator": 'gpu',
  "devices": 1,
  "num_workers": 48,
  "train_subset_size": 1400000,
  "validate_subset_size": 300000,
  "test_subset_size": 300000,
  "margin": 0
}

# Local MPS
# wandb.config = {
#   "model_name": 'TestData_StyleModel',
#   "batch_size": 32,
#   "learning_rate": 1e-4,
#   "max_length": 20,
#   "epochs": 1,
#   "precision": 16,
#   "accelerator": 'mps',
#   "devices": 1,
#   "num_workers": 8,
#   "train_subset_size": 700,
#   "validate_subset_size": 150,
#   "test_subset_size": 150,
#   "margin": 0
# }

wandb.init(project="contrastive_model", entity="commit_message_evaluation", config = wandb.config)

def load_data():
    #train = build_contrastive_pairs_data_dict('data/04a_Train_Set.pkl', cut_amount=369, subset_size=wandb.config['train_subset_size'])
    #validate = build_contrastive_pairs_data_dict('data/04b_Validate_Set.pkl', cut_amount=650, subset_size=wandb.config['validate_subset_size'])
    train = build_contrastive_pairs_data_dict('data/04c_Test_Set.pkl', cut_amount=647, subset_size=wandb.config['train_subset_size'])
    validate = build_contrastive_pairs_data_dict('data/04c_Test_Set.pkl', cut_amount=647, subset_size=wandb.config['validate_subset_size'])
    test = build_contrastive_pairs_data_dict('data/04c_Test_Set.pkl', cut_amount=647, subset_size=wandb.config['test_subset_size'])

    d = {
        'train': train,
        'validate': validate,
        'test': test
    }

    dataloader = DatasetDict(d)
    dataloader.set_format(type='pytorch')
    print("Tokenization (Runs 3x)")
    dataloader = dataloader.map(TokenizationWrapper(AutoTokenizer.from_pretrained(MODEL), wandb.config['max_length']).tokenize_function)
    return dataloader


class SBERT(L.LightningModule):
    def __init__(self, frozen = False, pretrained_weights = True):
        super().__init__()
        if pretrained_weights:
            self.sbert = AutoModel.from_pretrained(MODEL)
        else:
            config = BertConfig.from_pretrained(MODEL)
            self.sbert = AutoModel.from_config(config)
            for param in self.sbert.parameters():
                param.requires_grad = True
        if frozen:
            for param in self.sbert.parameters():
                param.requires_grad = False

    def forward(self, encoding):
        embeddings = self.sbert(**encoding)
        sentence_embeddings = mean_pooling(embeddings, encoding['attention_mask'])
        # Normalize embeddings
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings

loss_fn = nn.CosineEmbeddingLoss(margin=wandb.config['margin'])

class StyleModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.sbert_m_s = SBERT(frozen=True)
        self.sbert_m   = SBERT(pretrained_weights=False)

    def forward(self, encoding):
        embeddings_m_s = self.sbert_m_s(encoding)
        embeddings_m = self.sbert_m(encoding)
        # maybe use a feed-forward layer here instead
        embeddings_s = embeddings_m_s - embeddings_m
        return embeddings_s


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.sbert_m.parameters(), lr=wandb.config['learning_rate'])
        return optimizer

    def step(self, batch, batch_idx, step_prefix):
        encoding_1 = {}
        encoding_1['input_ids'] = batch['input_ids_1'].squeeze()
        encoding_1['token_type_ids'] = batch['token_type_ids_1'].squeeze()
        encoding_1['attention_mask'] = batch['attention_mask_1'].squeeze()

        encoding_2 = {}
        encoding_2['input_ids'] = batch['input_ids_2'].squeeze()
        encoding_2['token_type_ids'] = batch['token_type_ids_2'].squeeze()
        encoding_2['attention_mask'] = batch['attention_mask_2'].squeeze()

        # Compute prediction error
        X1_s = self(encoding_1)
        X2_s = self(encoding_2)

        # Compute Loss
        loss = loss_fn(X1_s, X2_s, batch['target'])
        
        # Log Loss to WandB
        wandb.log({f"{step_prefix}_loss": loss})

        self.log(f"{step_prefix}_loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        self.step(batch, batch_idx, 'validate')

    def test_step(self, batch, batch_idx):
        self.step(batch, batch_idx, 'test')


if __name__ == '__main__':
    data = load_data()
    
    train_dataloader = torch.utils.data.DataLoader(
        dataset=data['train'],
        batch_size=wandb.config['batch_size'],
        num_workers=wandb.config['num_workers'],
        shuffle=True,
        drop_last=True
    )

    validate_dataloader = torch.utils.data.DataLoader(
        dataset=data['validate'],
        batch_size=wandb.config['batch_size'],
        num_workers=wandb.config['num_workers'],
        drop_last=True
    )

    test_dataloader = torch.utils.data.DataLoader(
        dataset=data['test'],
        batch_size=wandb.config['batch_size'],
        num_workers=wandb.config['num_workers'],
        drop_last=True
    )

    model = StyleModel()
    wandb.watch(model)
    trainer = L.Trainer(
        accelerator=wandb.config['accelerator'],
        devices=wandb.config['devices'],
        max_epochs=wandb.config['epochs'],
        precision=wandb.config['precision']
    )
    trainer.fit(model, train_dataloader, validate_dataloader)
    trainer.test(model, test_dataloader)
    torch.save(model.state_dict(), wandb.config['model_name'])