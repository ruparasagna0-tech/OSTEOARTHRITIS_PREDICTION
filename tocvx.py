# File: combine_data.py
import pandas as pd
from scipy.io import mmread
import os

data_dir = r"C:\Users\sai12\OneDrive - Amrita vishwa vidyapeetham\bio_datasets"
samples = ["1", "3", "4"]

barcodes_files = [os.path.join(data_dir, f"barcodes{sample}.tsv") for sample in samples]
features_files = [os.path.join(data_dir, f"features{sample}.tsv") for sample in samples]
matrix_files = [os.path.join(data_dir, f"matrix{sample}.mtx") for sample in samples]

dataframes = []

for barcodes, features, matrix in zip(barcodes_files, features_files, matrix_files):
    barcodes_df = pd.read_csv(barcodes, header=None)
    features_df = pd.read_csv(features, sep="\t", header=None)
    sparse_matrix = mmread(matrix)
    dense_matrix = pd.DataFrame(sparse_matrix.toarray())
    
    # Add barcodes as columns and features as rows
    dense_matrix.columns = barcodes_df[0].values
    dense_matrix.index = features_df[1].values
    
    dataframes.append(dense_matrix)

# Concatenate all dataframes along columns
combined_matrix = pd.concat(dataframes, axis=1)
combined_matrix.to_csv('combined_data.csv', index=False)
