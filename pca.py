import numpy as np
from sklearn.decomposition import PCA

x = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components = 2)
pca.fit(x)

print(pca.explained_variance_ratio_)

T = pca.transform(x)
x.shape