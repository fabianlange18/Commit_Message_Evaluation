import pandas as pd
from tqdm import tqdm
from datasets import Dataset

def build_contrastive_pairs(data_path, cut_amount):
    train_data = pd.read_pickle(data_path)
    train_data_groups = train_data.groupby("author_email")

    positive_training_pairs = []

    for group in train_data_groups:
        for i, message_1 in enumerate(group[1]['message']):
            for message_2 in group[1]['message'].iloc[i+1:]:
                    positive_training_pairs.append([message_1, message_2, 1])

    groups_calculated = []

    negative_training_pairs = []

    for group in train_data_groups:
        groups_calculated.append(group[0])
        negative_groups = [group if group[0] not in groups_calculated else None for group in train_data_groups]
        negative_groups = list(filter(lambda item: item is not None, negative_groups))
        for message_1 in group[1]['message'].sample(n=cut_amount):
            for negative_group in negative_groups:
                for message_2 in negative_group[1]['message'].sample(n=cut_amount):
                    negative_training_pairs.append([message_1, message_2, -1])

    positive_training_pairs.extend(negative_training_pairs)
    return positive_training_pairs



def build_contrastive_pairs_data_dict(data_path, cut_amount, subset_size = 100_000_000):
    data = pd.read_pickle(data_path)
    train_data_groups = data.groupby("author_email")

    positive_count = 0
    negative_count = 0

    messages_1 = []
    messages_2 = []
    target     = []

    print("Setting up Contrastive Pairs (Runs 6x)")
    for group in tqdm(train_data_groups):
        for i, message_1 in enumerate(group[1]['message']):
            for message_2 in group[1]['message'].iloc[i+1:]:
                messages_1.append(message_1)
                messages_2.append(message_2)
                target.append(1)

    messages_1 = messages_1[:int(subset_size / 2)]
    messages_2 = messages_2[:int(subset_size / 2)]
    target = target[:int(subset_size / 2)]
    positive_count = len(messages_1)

    groups_calculated = []

    print("Setting up Contrastive Pairs (Runs 6x)")
    for group in tqdm(train_data_groups):
        groups_calculated.append(group[0])
        negative_groups = [group if group[0] not in groups_calculated else None for group in train_data_groups]
        negative_groups = list(filter(lambda item: item is not None, negative_groups))
        for message_1 in group[1]['message'].sample(n=cut_amount):
            for negative_group in negative_groups:
                for message_2 in negative_group[1]['message'].sample(n=cut_amount):
                    messages_1.append(message_1)
                    messages_2.append(message_2)
                    target.append(-1)
                    negative_count += 1

    messages_1 = messages_1[:subset_size]
    messages_2 = messages_2[:subset_size]
    target = target[:subset_size]
    negative_count = len(messages_1) - positive_count

    print("Positive Pairs:")
    print(positive_count)
    print("Negative Pairs:")
    print(negative_count)
    return Dataset.from_dict({'messages_1': messages_1, 'messages_2': messages_2, 'target': target})