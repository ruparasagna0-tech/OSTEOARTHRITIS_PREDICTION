# labels.py
import pandas as pd

def create_labels(samples, combined_data):
    # Initialize an empty list to store labels
    labels = []
    
    # Loop over the samples and assign labels based on suffix
    for sample in samples:
        if sample == "1":
            labels.append([1] * combined_data.filter(like=sample).shape[1])
        elif sample == "3":
            labels.append([2] * combined_data.filter(like=sample).shape[1])
        elif sample == "4":
            labels.append([3] * combined_data.filter(like=sample).shape[1])
    
    # Flatten the list of lists into a single list
    labels = [label for sublist in labels for label in sublist]
    
    # Convert labels to a pandas Series
    labels = pd.Series(labels)
    
    return labels
