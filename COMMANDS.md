## Commands that would be useful for datascience

### Useful file inclusions
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

### File Handling
##### 1. Open a csv file
```python
data = pd.read_csv('file.csv')
```
##### 2. List files in directory
```python
from subprocess import check_output
print(check_output(["ls", "folder"]).decode("utf8"))
```

### Plots (Matplotlib and Seaborn)
##### 1. Correlation HeatMap
```python
f, ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt=".1f", ax=ax)
```

#### 2. Line Plot
```python
data.Speed.plot(kind='line', color='g', label='Attack', linewidth=1, alpha=0.5, grid=True, linestyle=':')
data.Defense.plot(color='r', label='Defense', linewidth=1, alpha=0.5, grid=True, linestyle='-.')
plt.legend(loc = 'upper right')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Line Plot')
```

#### 3. Scatter Plot
```python
data.plot(kind='scatter', x='Attack', y='Defense', alpha=0.5, color='red')
plt.xlabel('Attack')
plt.ylabel('Defense')
plt.title('Attack Defense Scatter Plot')
```

#### 4. Histogram
```python
data.Speed.plot(kind='hist', bins=50, figsize=(15, 15))
```

#### 5. Boxplot
```python
data.boxplot(column='Attack', by='Legendary')
```

#### 6. Subplots
```python
data.plot(subplots=True)
```

### Clean Data
#### 1. Tidy Data
```python
melted = pd.melt(frame=data, id_vars='Name', value_vars=['Attack', 'Defense'])
```

#### 2. Pivoting Data
```python
melted.pivot(index='Name', columns='variable', values='value')
```

### 3. Concatenating Data
```python
data1 = data.head()
data2= data.tail()
conc_data_row = pd.concat([data1,data2],axis=0, ignore_index=True)
```

#### 4. Info about dataset
```python
print(data.info())      # info
print(data.head())      # first 5 rows
print(data.tail())      # last 5 rows
print(data.columns)     # no. of features
print(data.shape)       # no. of rows and columns
print(data['Type_2'].value_counts(dropna=False))
print(data.describe())
print(data.dtypes)      # datatypes
```

### Manipulate Data Frames
#### 1. Indexing
```python
data = data.set_index('#')
print(data['HP'][1])
print(data.HP[1])
print(data.loc[1, ['HP']])
print(data[['HP', 'Attack']])
```

#### 2. Slicing
```python
print(type(data["HP"]))     # series
print(type(data[["HP"]]))   # data frames
print(data.loc[1:10,"HP":"Defense"])
print(data.loc[1:10,"Speed":])
```

#### 3. Filtering pandas data frame
```python
x = [data['Defense'] > 200]
x = [np.logical_and(data['Defense']>200, data['Attack']>100)]
x = [(data['Defense']>200) & (data['Attack']>100)]
```

#### 4. Transforming Data
```python
data.HP.apply(lambda n : n/2)
data["total_power"] = data.Attack + data.Defense
```

#### 5. Hierarchical Indexing
```python
data1 = data.set_index(['Type_1', 'Type_2'])
```