import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from sklearn.cluster import KMeans

matplotlib.style.use('ggplot')

mat = sio.loadmat('datasets/face.mat')
df = pd.DataFrame(mat['images'])

kmeans = KMeans(n_clusters = 5)
kmeans.fit(df)

labels = kmeans.predict(df)
centroids = kmeans.cluster_centers_

print(centroids)