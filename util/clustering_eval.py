import numpy as np
import pandas as pd
from collections import Counter

import warnings
warnings.filterwarnings('ignore')

def clustering_summary(predictions, n_clusters, data):
    clustering_summary = pd.DataFrame(columns=['Number of Messages', 'Number of different Authors', 'Average number of commits per different Author', 'Most common Author', 'Number of different Projects', 'Average number of commits per different Project', 'Most common project'])
    clustering_summary['Number of Messages'] = [tuple[1] for tuple in sorted(Counter(predictions).items(), key=lambda pair: pair[0])]
    for label in range(n_clusters):
        author_emails = []
        projects = []
        for i, _ in enumerate(data['message']):
            if predictions[i] == label:
                author_emails.append(data["author_email"][i])
                projects.append(data["project"][i])
        commiter_emails_count = Counter(author_emails)
        projects_count = Counter(projects)
        clustering_summary['Number of different Authors'][label] = int(len(commiter_emails_count))
        clustering_summary['Average number of commits per different Author'][label] = float(np.mean(list(commiter_emails_count.values())))
        clustering_summary['Most common Author'][label] = commiter_emails_count.most_common(1)[0]
        clustering_summary['Number of different Projects'][label] = int(len(projects_count))
        clustering_summary['Average number of commits per different Project'][label] = float(np.mean(list(projects_count.values())))
        clustering_summary['Most common project'][label] = projects_count.most_common(1)[0]
        convert_dict = {
            'Number of Messages' : float,
            'Number of different Authors' : float,
            'Average number of commits per different Author' : float,
            'Most common Author' : str,
            'Number of different Projects' : float,
            'Average number of commits per different Project' : float,
            'Most common project' : str
            }
        clustering_summary = clustering_summary.astype(convert_dict)
    print(f"There are {len(data['author_email'].unique())} different authors.")
    print(f"There are {len(data['project'].unique())} different projects.")
    return clustering_summary





def print_clustering_classes(predictions, n_clusters, data, message_details=False):
    for label in range(n_clusters):
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