import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

mat = sio.loadmat('datasets/face.mat')
X = mat['images']
X = scale(X)

pca = PCA(n_components=64)
pca.fit(X)

var = pca.explained_variance_ratio_
var1 = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

print (var1)
plt.plot(var1)