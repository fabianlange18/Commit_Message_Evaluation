# Train a first Transformer Model
# Tutorial:
# https://huggingface.co/docs/transformers/training

MODEL = 'bert-base-cased'

import evaluate
import numpy as np
import pandas as pd
from datasets import Dataset
from datasets.dataset_dict import DatasetDict
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

train_data = pd.read_pickle('../../data/05a_Authors_Train_Set.pkl')
validate_data = pd.read_pickle('../../data/05b_Authors_Validate_Set.pkl')
test_data = pd.read_pickle('../../data/05c_Authors_Test_Set.pkl')

d = {
    'train': Dataset.from_dict({'label': train_data['label'].astype('int32'), 'text': train_data['message']}),
    'val': Dataset.from_dict({'label': validate_data['label'].astype('int32'), 'text': validate_data['message']}),
    'test': Dataset.from_dict({'label': test_data['label'].astype('int32'), 'text': test_data['message']})
}
dataset_dict = DatasetDict(d)

model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=42)
tokenizer = AutoTokenizer.from_pretrained(MODEL)

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["val"].shuffle(seed=42).select(range(1000))

metric = evaluate.load('accuracy')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(output_dir='test_trainer_small', evaluation_strategy='epoch')

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics
)

trainer.train()