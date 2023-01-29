import torch

class TokenizationWrapper():

    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def tokenize_function(self, examples):
        all_features = {}
        features_1 = self.tokenizer(examples['messages_1'], padding='max_length', truncation=True, return_tensors='pt', max_length=self.max_length)
        features_2 = self.tokenizer(examples['messages_2'], padding='max_length', truncation=True, return_tensors='pt', max_length=self.max_length)
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