import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA

matplotlib.style.use('ggplot')
pokemon = pd.read_csv('datasets/pokemon.csv')
df = pokemon[['HP', 'Attack', 'Defense', 'Sp_Atk', 'Sp_Def', 'Speed']]

#Describe pca perimeters
#print (df.describe())

pca = PCA(n_components=2, svd_solver='full')
pca.fit(df)
T = pca.transform(df)
#print(df.shape)
#print(T.shape)

#Explained variance ratio
#print(pca.explained_variance_ratio_)

#Correlation between components
#components = pd.DataFrame(pca.components_, columns=df.columns, index=[1, 2])

def get_important_features(transformed_features, components_, columns):
    num_columns = len(columns)
    xvector = components_[0] * max(transformed_features[:, 0])
    yvector = components_[1] * max(transformed_features[:, 1])
    
    important_features = { columns[i] : math.sqrt(xvector[i]**2 + yvector[i]**2) for i in range(num_columns) }
    important_features = sorted(zip(important_features.values(), important_features.keys()), reverse=True)
    print ("Features by importance:\n", important_features)
    
def draw_vectors(transformed_features, components_, columns):
    num_columns = len(columns)
    xvector = components_[0] * max(transformed_features[:, 0])
    yvector = components_[1] * max(transformed_features[:, 1])
    
    ax = plt.axes()
    for i in range(num_columns):
        plt.arrow(0, 0, xvector[i], yvector[i], color='b', width=0.0005, head_width=0.02, alpha=0.75)
        plt.text(xvector[i]*1.2, yvector[i]*1.2, list(columns)[i], color='b', alpha=0.75)
        
    return ax
    
#get_important_features(T, pca.components_, df.columns.values)

ax = draw_vectors(T, pca.components_, df.columns.values)
T_df = pd.DataFrame(T)
T_df.columns = ['component1', 'component2']

T_df['color'] = 'y'
T_df.loc[T_df['component1'] > 125, 'color'] = 'g'
T_df.loc[T_df['component2'] > 125, 'color'] = 'r'

plt.xlabel('Principle Component 1')
plt.ylabel('Principle Component 2')
plt.scatter(T_df['component1'], T_df['component2'], color=T_df['color'], alpha=0.5)
plt.show()

# High Attack, High Sp. Atk, all of these pokemon are legendary
#print(pokemon.loc[T_df[T_df['color'] == 'g'].index])

# High Defense, Low Speed
#print(pokemon.loc[T_df[T_df['color'] == 'r'].index])