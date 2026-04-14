# preprocess.py
import pandas as pd
from scipy.io import mmread
import os

def load_data(data_dir, samples):
    # Initialize an empty list to store dataframes
    dataframes = []
    
    for sample in samples:
        barcodes_file = os.path.join(data_dir, f"barcodes{sample}.tsv")
        features_file = os.path.join(data_dir, f"features{sample}.tsv")
        matrix_file = os.path.join(data_dir, f"matrix{sample}.mtx")

        # Load barcodes and features
        barcodes_df = pd.read_csv(barcodes_file, header=None)
        features_df = pd.read_csv(features_file, sep="\t", header=None)

        # Load matrix and convert to dense
        sparse_matrix = mmread(matrix_file)
        dense_matrix = pd.DataFrame(sparse_matrix.toarray())

        # Assign barcodes and features as row/column names
        dense_matrix.columns = barcodes_df[0].values  # Barcodes as columns
        dense_matrix.index = features_df[1].values    # Gene names as rows

        # Append the dense matrix to the list of dataframes
        dataframes.append(dense_matrix)

    # Concatenate all dataframes along columns (axis=1)
    combined_matrix = pd.concat(dataframes, axis=1)

    # Return combined data as a DataFrame
    return combined_matrix
