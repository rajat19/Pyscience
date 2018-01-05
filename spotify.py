import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# from subprocess import check_output
# print(check_output(["ls", "datasets/spotify"]).decode("utf8"))
df = pd.read_csv('datasets/spotify/featuresdf.csv')
small = df.drop(['name', 'artists'], axis=1)

# How danceability affected ranking
x = small.danceability
# sns.distplot(x)
# sns.distplot(x, kde=False, bins=20, rug=True)
# sns.distplot(x, hist=False, rug=True)
# sns.kdeplot(x, shade=True, label="danceability")

# How energy affected ranking
# sns.distplot(small.energy)
# sns.kdeplot(small.energy, label="energy")
plt.show()