import spacy
import numpy as np
import pandas as pd
from collections import Counter
from spacytextblob.spacytextblob import SpacyTextBlob


import warnings
warnings.filterwarnings('ignore')

def clustering_summary(predictions, data):
    if np.isin(-1, predictions):
        predictions += 1
    clustering_summary = pd.DataFrame(columns=['Number of Messages', 'Number of different Authors', 'Median number of commits per different Author', 'Most common Author', 'Number of different Projects', 'Median number of commits per different Project', 'Most common project'])
    clustering_summary['Number of Messages'] = [tuple[1] for tuple in sorted(Counter(predictions).items(), key=lambda pair: pair[0])]
    for label in clustering_summary.index:
        author_emails = []
        projects = []
        for i, _ in enumerate(data['message']):
            if predictions[i] == label:
                author_emails.append(data["author_email"][i])
                projects.append(data["project"][i])
        commiter_emails_count = Counter(author_emails)
        projects_count = Counter(projects)
        clustering_summary['Number of different Authors'][label] = int(len(commiter_emails_count))
        clustering_summary['Median number of commits per different Author'][label] = float(np.median(list(commiter_emails_count.values())))
        clustering_summary['Most common Author'][label] = commiter_emails_count.most_common(1)[0]
        clustering_summary['Number of different Projects'][label] = int(len(projects_count))
        clustering_summary['Median number of commits per different Project'][label] = float(np.median(list(projects_count.values())))
        clustering_summary['Most common project'][label] = projects_count.most_common(1)[0]
        convert_dict = {
            'Number of Messages' : float,
            'Number of different Authors' : float,
            'Median number of commits per different Author' : float,
            'Most common Author' : str,
            'Number of different Projects' : float,
            'Median number of commits per different Project' : float,
            'Most common project' : str
            }
        clustering_summary = clustering_summary.astype(convert_dict)
    print(f"There are {len(data['author_email'].unique())} different authors.")
    print(f"There are {len(data['project'].unique())} different projects.")
    return clustering_summary





def clustering_spacy_evaluation(predictions, data):
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe('spacytextblob')

    spacy_summary = pd.DataFrame(
        columns=[str(label) for label in sorted(list(set(predictions)))],
        index=['length_mean', 'length_std', 'n_uppercase_mean', 'n_uppercase_std', 'polarity_mean', 'polarity_std', 'subjectivity_mean', 'subjectivity_std'])

    if np.isin(-1, predictions):
        predictions += 1
    
    for label in sorted(list(set(predictions))):
        messages = data['message'].where(predictions == label)
        messages.dropna(inplace=True)
        messages.reset_index(drop=True, inplace=True)
        docs = nlp.pipe(messages)
        
        lengths = []
        n_upper_case_letters = []
        polarities = []
        subjectivities = []

        for message in messages:
            lengths.append(len(message))
            n_upper_case_letters.append(sum(1 for c in message if c.isupper()))

        for doc in docs:
            polarities.append(doc._.blob.polarity)
            subjectivities.append(doc._.blob.subjectivity)

        spacy_summary[str(label)]['length_mean'] = np.mean(lengths)
        spacy_summary[str(label)]['length_std'] = np.std(lengths)
        spacy_summary[str(label)]['n_uppercase_mean'] = np.mean(n_upper_case_letters)
        spacy_summary[str(label)]['n_uppercase_std'] = np.std(n_upper_case_letters)
        spacy_summary[str(label)]['polarity_mean'] = np.mean(polarities)
        spacy_summary[str(label)]['polarity_std'] = np.std(polarities)
        spacy_summary[str(label)]['subjectivity_mean'] = np.mean(subjectivities)
        spacy_summary[str(label)]['subjectivity_std'] = np.std(subjectivities)

    spacy_summary.round(2)

    return spacy_summary.style.apply(lambda s: ['background-color: white' if x%4<2 else 'background-color: lightgrey' for x in range(len(s))])





def print_clustering_classes(predictions, data, message_details=False):
    if np.isin(-1, predictions):
        predictions += 1
    for label in sorted(list(set(predictions))):
        print("\n________________ Class " + str(label) + " ________________\n")
        class_counter = 1
        author_emails = []
        projects = []
        print_examples = True
        for i, message in enumerate(data["message"]):
            if class_counter == 11:
                print_examples = False
            if predictions[i] == label:
                author_emails.append(data["author_email"][i])
                projects.append(data["project"][i])
                if print_examples:
                    print("___")
                    print(str(class_counter) + ") ")
                    print(message)
                    if message_details:
                        print()
                        print("- - - ")
                        print("Committer: " + str(data["author_email"][i]))
                        print("Project:   " + str(data["project"][i]))
                class_counter += 1
        print("_________________")
        print()
        print("Number of messages in this class: " + str(class_counter - 1))
        print("Most common author:")
        print(Counter(author_emails).most_common(1)[0])
        print("Most common project:")
        print(Counter(projects).most_common(1)[0])
        print()
        print()