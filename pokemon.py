import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('datasets/pokemon/pokemon.csv')

# Line Plot
data.Speed.plot(
	kind = 'line',
	color = 'g',
	label = 'Attack',
	linewidth = 1,
	alpha = 0.5,
	grid = True,
	linestyle = ':'
)
data.Defense.plot(
	color = 'r',
	label = 'Defense',
	linewidth = 1,
	alpha = 0.5,
	grid = True,
	linestyle = '-.'
)
plt.legend(loc = 'upper right')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Line Plot')

# Scatter Plot
data.plot(
	kind = 'scatter',
	x = 'Attack',
	y = 'Defense',
	alpha = 0.5,
	color = 'red'
)
plt.xlabel('Attack')
plt.ylabel('Defense')
plt.title('Attack Defense Scatter Plot')

# Histogram
# bins = no. of bar in figure
data.Speed.plot(
	kind = 'hist',
	bins = 50,
	figsize = (15, 15)
)

# List comprehension
# threshold = sum(data.Speed) / len(data.Speed)
# data['speed_level'] = ['high' if i > threshold else 'low' for i in data.Speed]
# print (data.loc[:10, ['speed_level', 'Speed']])

# print(data['Type_2'].value_counts(dropna=False))
print(data.describe())

# Boxplot
data.boxplot(column='Attack', by='Legendary')

data_new = data.head()
melted = pd.melt(frame=data_new, id_vars='Name', value_vars=['Attack', 'Defense'])
print(melted.pivot(index='Name', columns='variable', values='value'))

# print(data.info())
# print(data['Type_2'].value_counts(dropna=False))
# data1 = data
# data1['Type_2'].dropna(inplace=True)
# assert data['Type_2'].notnull().all()
# data['Type_2'].fillna('empty', inplace=True)
# assert data['Type_2'].notnull().all()

# data1 = data.loc[:, ["Attack", "Defense", "Speed"]]
# data1.plot()
# data1.plot(subplots=True)
# data1.plot(kind = "scatter",x="Attack",y = "Defense")
# data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True)
# fig, axes = plt.subplots(nrows=2,ncols=1)
# data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True,ax = axes[0])
# data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative = True)

x = data['Generation'] == 1
old = data[x].set_index(['#'])
old = old[~old.index.duplicated(keep='first')]
sort = old.sort_values(by='Total', ascending=False)
categorised = sort.set_index(['Type_1', 'Type_2'])
print(categorised.head(50)[['Name', 'Legendary']])

plt.show()