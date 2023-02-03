import pandas as pd
import random
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

    positive_count_total = len(messages_1)

    # Take the subset
    c = list(zip(messages_1, messages_2, target))
    final_messages_1 = []
    final_messages_2 = []
    final_target = []
    for message_1, message_2, target in random.sample(c, int(subset_size / 2)):
        final_messages_1.append(message_1)
        final_messages_2.append(message_2)
        final_target.append(target)
    
    positive_count_subset = len(final_messages_1)

    print("Positive Pairs:")
    print(f"{positive_count_subset} out of a total of {positive_count_total}")

    groups_calculated = []
    messages_1 = []
    messages_2 = []
    target     = []

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

    negative_count_total = len(messages_1)
    
    c = list(zip(messages_1, messages_2, target))
    for message_1, message_2, target in random.sample(c, int(subset_size / 2)):
        final_messages_1.append(message_1)
        final_messages_2.append(message_2)
        final_target.append(target)

    negative_count_subset = len(final_messages_1) - positive_count_subset

    print("Negative Pairs:")
    print(f"{negative_count_subset} out of a total of {negative_count_total}")
    return Dataset.from_dict({'messages_1': final_messages_1, 'messages_2': final_messages_2, 'target': final_target})