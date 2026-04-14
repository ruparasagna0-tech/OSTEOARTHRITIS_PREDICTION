from sklearn.decomposition import SparsePCA
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from preprocess import scaled_data
# Use Sparse PCA for better results with sparse data
sparse_pca = SparsePCA(n_components=2)
sparse_pca_result = sparse_pca.fit_transform(scaled_data)

# Plot the first two sparse principal components
plt.figure(figsize=(8, 6))
plt.scatter(sparse_pca_result[:, 0], sparse_pca_result[:, 1], c='blue', label='OA Samples', alpha=0.5)
plt.xlabel('Sparse PC 1')
plt.ylabel('Sparse PC 2')
plt.title('Sparse PCA of Gene Expression Data')
plt.show()
