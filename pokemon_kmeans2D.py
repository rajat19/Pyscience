import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans

plt.style.use('ggplot')

pokemon = pd.read_csv('datasets/pokemon.csv')
df = pokemon[['Total', 'Attack', 'Defense', 'HP']]
kmeans = KMeans(n_clusters=3)
kmeans.fit(df)
labels = kmeans.labels_

df.plot.scatter(x='Total', y='Defense', edgecolor='k', c=labels.astype(np.float))