import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

plt.style.use('ggplot')

pokemon = pd.read_csv('datasets/pokemon.csv')
df = pokemon[['Total', 'Attack', 'HP']]
kmeans = KMeans(n_clusters=5)
kmeans.fit(df)
labels = kmeans.labels_

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Total')
ax.set_ylabel('Attack')
ax.set_zlabel('HP')

ax.scatter(df.Total, df.Attack, df.HP, edgecolor='k', c=labels.astype(np.float))

plt.show()