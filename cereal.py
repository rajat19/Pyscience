import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('datasets/cereals/cereal.csv')
mfr_mapping = {
    'A': 'American Home Food',
    'G': 'General Mills',
    'K': 'Kelloggs',
    'N': 'Nabisco',
    'P': 'Post',
    'Q': 'Quaker Oats',
    'R': 'Ralston Purina',
}
df['mapped_mfr'] = df['mfr'].map(mfr_mapping)
# A = American Home Food Products;
# G = General Mills
# K = Kelloggs
# N = Nabisco
# P = Post
# Q = Quaker Oats
# R = Ralston Purina

# sns.jointplot(x="rating", y="calories", data=df)
# sns.jointplot(x="rating", y="protein", data=df)
# sns.jointplot(x="rating", y="fat", data=df)
# sns.jointplot(x="rating", y="sodium", data=df)
# sns.jointplot(x="rating", y="fiber", data=df)
# sns.jointplot(x="rating", y="carbo", data=df)
# sns.jointplot(x="rating", y="sugars", data=df)
# sns.jointplot(x="rating", y="potass", data=df)
# sns.jointplot(x="rating", y="vitamins", data=df)

# sns.violinplot(y="mapped_mfr", x="rating", hue="type", jitter=True, data=df, split=True, palette="Set3", inner="stick")
# g = sns.PairGrid(
#     df,
#     x_vars=["calories", "protein", "fat"],
#     y_vars=["rating"],
#     aspect=0.75,
#     size=3.5
# )
# g.map(sns.jointplot)

sns.factorplot(x="rating", y="mapped_mfr", hue="type", data=df, kind="swarm")
plt.show()