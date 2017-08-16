import pandas as pd
import matplotlib

matplotlib.style.use('ggplot')

stud = pd.read_csv('datasets/students.data', index_col=0)

series = stud.G3
dataframe = stud[['G3', 'G2', 'G1']]

series.plot.hist(alpha=0.5)
stud.plot.scatter(x='G1', y='G2')