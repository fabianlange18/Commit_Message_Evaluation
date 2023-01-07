import pandas as pd

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