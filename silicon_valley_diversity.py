import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('datasets/silicon_valley_diversity/dataset.csv')
totals = df[df['job_category'] == 'Totals']
totals['count'] = totals['count'].astype(dtype='int32')
# company,year,race,gender,job_category,count
# print(df['company'].unique())
# print(df['race'].unique())
# print(df['job_category'].unique())
# print(df[df['company'] == 'Adobe'][['race', 'gender', 'count']])

gender = totals.groupby(['company', 'gender'])['count'].sum().reset_index()
companies = list(gender['company'].unique())
companies_total = list(gender.groupby(['company'])['count'].sum())

# for i, company in enumerate(companies):
#     count_female = gender.at[gender[gender['company'] == company].index[0], 'count']
#     count_male = gender.at[gender[gender['company'] == company].index[1], 'count']
#     print('{}:'.format(company))
#     print('\tProportions of men: {:0.2f}%'.format(100*float(count_male)/companies_total[i]))
#     print('\tProportions of women: {:0.2f}%'.format(100*float(count_female)/companies_total[i]))

# plt.figure(figsize=(15,30))
# sns.barplot(x='count', y='company',hue='gender',data=gender)

# overall = df[df['race']=='Overall_totals']
# overall['count'] = overall['count'].str.replace('na','0').astype(float)
# sns.factorplot(x='count',y='company',col='job_category',kind='point',col_wrap=4,data=overall)

original_races = df[df['race'] != 'Overall_totals']
# original_races = original_races[original_races['race'] != 'Two_or_more_races']
original_races['count'] = original_races['count'].str.replace('na','0').astype(float)

sns.factorplot(x='count',y='company',col='race',kind='point',col_wrap=4,data=original_races)
# print(original_races.head())
plt.show()