import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from ggplot import *

data = pd.read_csv('datasets/pokemon/pokemon.csv')
combats = pd.read_csv('datasets/pokemon/combats.csv')
tests = pd.read_csv('datasets/pokemon/tests.csv')

# data = data.drop(['Name'], axis=1)

# More powerful if legendary
# data['Legendary'] = data['Legendary'].map( {False: 0, True: 1} )

# map types
# type_mapping = {
#     'Grass': 1,
#     'Fire': 2,
#     'Water': 3,
#     'Bug': 4,
#     'Normal': 5,
#     'Poison': 6,
#     'Electric': 7,
#     'Ground': 8,
#     'Fairy': 9,
#     'Fighting': 10,
#     'Psychic': 11,
#     'Rock': 12,
#     'Ghost': 13,
#     'Ice': 14,
#     'Dragon': 15,
#     'Dark': 16,
#     'Steel': 17,
#     'Flying': 18,
# };

# data['Type 1'] = data['Type 1'].map(type_mapping)
# data['Type 2'] = data['Type 2'].map(type_mapping)
data['Type 2'] = data['Type 2'].fillna('None')
# data['Type 2'] = data['Type 2'].astype(int)

battles = combats.apply(lambda x: x.value_counts()).fillna(0).astype(int)
battles['Fights'] = battles['First_pokemon'] + battles['Second_pokemon']
battles['Win_per'] = battles['Winner'] / battles['Fights']
battles['id'] = battles.index
data.set_index('#')
merged = pd.concat([data, battles], axis=1)
merged = data.merge(battles, left_on='#', right_on='id', how='left')
merged = merged.fillna(0)

print(merged.info())
# print(merged.sort_values(by='First_pokemon', ascending=False).head())
tests['Winner'] = 0
np_tests = tests.as_matrix()
for x in np_tests:
    x[2] = x[0] if merged['Win_per'][x[0] - 1] > merged['Win_per'][x[1] - 1] else x[1]

print(tests.head())

tests.to_csv('datasets/pokemon/result.csv', sep=",", index=False)
plt.show()