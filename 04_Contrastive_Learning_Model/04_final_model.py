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

import sys
sys.path.append('.')
from util.contrastive_pairs import build_contrastive_pairs_data_dict

# Server GPU
wandb.config = {
  "batch_size": 256,
  "learning_rate": 1e-3,
  "max_length": 20,
  "epochs": 25,
  "precision": 16,
  "accelerator": 'gpu',
  "devices": 1,
  "train_subset_size": 70000,
  "validate_subset_size": 15000
}

# Local MPS
# wandb.config = {
#   "batch_size": 32,
#   "learning_rate": 1e-3,
#   "max_length": 20,
#   "epochs": 10,
#   "precision": 16,
#   "accelerator": 'mps',
#   "devices": 1,
#   "train_subset_size": 65,
#   "validate_subset_size": 35
# }

wandb.init(project="contrastive_model", entity="commit_message_evaluation", config = wandb.config)

def load_data():
    train = build_contrastive_pairs_data_dict('data/04a_Train_Set.pkl', 1) # , 369)
    validate = build_contrastive_pairs_data_dict('data/04b_Validate_Set.pkl', 1) #, 650)
    test = build_contrastive_pairs_data_dict('data/04c_Test_Set.pkl', 1) # 647)

    d = {
        'train': train,
        'validate': validate,
        'test': test
    }

    dataloader = DatasetDict(d)
    dataloader.set_format(type='pytorch')
    print("Tokenization (Runs 3x)")
    dataloader = dataloader.map(tokenize_function)
    return dataloader

tokenizer = AutoTokenizer.from_pretrained(MODEL)

def tokenize_function(examples):
    all_features = {}
    features_1 = tokenizer(examples['messages_1'], padding='max_length', truncation=True, return_tensors='pt', max_length=wandb.config['max_length'])
    features_2 = tokenizer(examples['messages_2'], padding='max_length', truncation=True, return_tensors='pt', max_length=wandb.config['max_length'])
    all_features['input_ids_1']      = features_1['input_ids']
    all_features['token_type_ids_1'] = features_1['token_type_ids']
    all_features['attention_mask_1'] = features_1['attention_mask']
    all_features['input_ids_2']      = features_2['input_ids']
    all_features['token_type_ids_2'] = features_2['token_type_ids']
    all_features['attention_mask_2'] = features_2['attention_mask']
    return all_features

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


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

loss_fn = nn.CosineEmbeddingLoss()

class StyleModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.sbert_m_s = SBERT(pretrained_weights=False)
        self.sbert_m   = SBERT(frozen=True)

    def forward(self, encoding):
        embeddings_m_s = self.sbert_m_s(encoding)
        embeddings_m = self.sbert_m(encoding)
        # maybe use a feed-forward layer here instead
        embeddings_s = embeddings_m_s - embeddings_m
        return embeddings_s


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.sbert_m_s.parameters(), lr=wandb.config['learning_rate'])
        return optimizer

    def training_step(self, batch, batch_idx):
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
        wandb.log({"train_loss": loss})

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
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
        wandb.log({"val_loss": loss})
        self.log('val_loss', loss, batch_size=wandb.config['batch_size'])


# class ContrastivePairsData(L.LightningDataModule):
#     def __init__(self, batch_size):
#         super().__init__()
#         self.batch_size = batch_size

#     def prepare_data(self):
#         load_data()

#     def train_dataloader(self):
#         return torch.utils.data.DataLoader(
#             dataset=data['train'][:wandb.config['subset_size']],
#             batch_size=wandb.config['batch_size'],
#             shuffle=True
#         )

if __name__ == '__main__':
    data = load_data()
    
    train_subset_indices = random.choices(list(range(0, len(data['train']))), k=wandb.config['train_subset_size'])
    train_subset = torch.utils.data.Subset(data['train'], train_subset_indices)

    validate_subset_indices = random.choices(list(range(0, len(data['validate']))), k=wandb.config['validate_subset_size'])
    validate_subset = torch.utils.data.Subset(data['validate'], validate_subset_indices)
    
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_subset,
        batch_size=wandb.config['batch_size'],
        shuffle=True,
        drop_last=True
    )

    validate_dataloader = torch.utils.data.DataLoader(
        dataset=validate_subset,
        batch_size=wandb.config['batch_size']
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
    torch.save(model.state_dict(), 'model/Subset_Style_Model.pt')