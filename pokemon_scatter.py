import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

matplotlib.style.use('ggplot')
df = pd.read_csv('datasets/pokemon.csv', index_col=0)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Total')
ax.set_ylabel('Attack')
ax.set_zlabel('HP')
ax.scatter(df.Total, df.Attack, df.HP, c='r', marker='o')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Total')
ax.set_ylabel('Defense')
ax.set_zlabel('HP')
ax.scatter(df.Total, df.Defense, df.HP, c='b', marker='o')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Total')
ax.set_ylabel('Speed')
ax.set_zlabel('HP')
ax.scatter(df.Total, df.Speed, df.HP, c='g', marker='o')
plt.show()