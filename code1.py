# combined.py
from preprocess import load_data
from labels import create_labels
from random_forest_model import run_random_forest_model
import os
import pandas as pd

# Define your data directory and sample names
data_dir = r"C:\Users\sai12\OneDrive - Amrita vishwa vidyapeetham\bio_datasets"
samples = ["1", "3", "4"]

# Load and combine data (barcodes, features, and matrices)
combined_matrix = load_data(data_dir, samples)

# Check if combined_matrix is a DataFrame
if isinstance(combined_matrix, pd.DataFrame):
    print("Data loaded successfully as DataFrame.")
else:
    print("Error: combined_matrix is not a DataFrame.")

# Create labels based on the sample file suffixes
labels = create_labels(samples, combined_matrix)

# Now you have the combined data matrix and the corresponding labels
print(f"Combined data shape: {combined_matrix.shape}")
print(f"Labels shape: {labels.shape}")

# Run the Random Forest Model with 10-fold cross-validation
run_random_forest_model(combined_matrix, labels)
