import torch
from transformers import AutoTokenizer, AutoModel

MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

######### Helper functions #########
# Tokenization
tokenizer = AutoTokenizer.from_pretrained(MODEL)

def tokenize_function(examples, max_length = 25):
    return tokenizer(examples, padding=True, truncation=True, return_tensors='pt', max_length=max_length)

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

######### End Helper functions #########


class SBERT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sbert = AutoModel.from_pretrained(MODEL)

    def forward(self, sentence): # sentence
        encoding = tokenize_function(sentence)
        encoding['attention_mask'] = encoding['attention_mask']
        encoding['input_ids'] = encoding['input_ids']
        encoding['token_type_ids'] = encoding['token_type_ids']

        embeddings = self.sbert(**encoding)
        # Perform pooling
        sentence_embeddings = mean_pooling(embeddings, encoding['attention_mask'])
        # Normalize embeddings
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings


class StyleModel(torch.nn.Module):
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