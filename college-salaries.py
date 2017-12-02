import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# from subprocess import check_output
# print(check_output(["ls", "datasets/college-salaries"]).decode("utf8"))
degrees = pd.read_csv('datasets/college-salaries/degrees-that-pay-back.csv')
sal_by_type = pd.read_csv('datasets/college-salaries/salaries-by-college-type.csv')
sal_by_region = pd.read_csv('datasets/college-salaries/salaries-by-region.csv')

degrees.columns = ['major','bgn_p50','mid_p50','delta_bgn_mid','mid_p10','mid_p25','mid_p75','mid_p90']
sal_by_type.columns = ['school', 'type', 'bgn_p50','mid_p50','mid_p10','mid_p25','mid_p75','mid_p90']

# print(len(degrees['major'].unique()))
# print(degrees.sort_values(by="bgn_p50", ascending=False).head(20))
# print(sal_by_type.sort_values(by="Starting Median Salary", ascending=False)[['School Name', 'School Type', 'Starting Median Salary']].head(20))
# print(sal_by_type.describe())
# print(degrees.sort_values(by="delta_bgn_mid", ascending=False).head(20))

dollar_cols = ['bgn_p50','mid_p50','mid_p10','mid_p25','mid_p75','mid_p90']
for x in dollar_cols:
    degrees[x] = degrees[x].str.replace("$", "")
    degrees[x] = degrees[x].str.replace(",", "")
    degrees[x] = pd.to_numeric(degrees[x])
    sal_by_type[x] = sal_by_type[x].str.replace("$", "")
    sal_by_type[x] = sal_by_type[x].str.replace(",", "")
    sal_by_type[x] = pd.to_numeric(sal_by_type[x])

# print(degrees.head())
# print(degrees.describe())

degrees = degrees.sort_values(by="bgn_p50", ascending=False)
degrees = degrees.reset_index()

fig = plt.figure(figsize=(8, 12))
y = len(degrees.index) - degrees.index
x = degrees['bgn_p50']
x2 = degrees['mid_p50']
labels = degrees['major']

plt.scatter(x, y, color='g', label='Starting Median Salary')
plt.yticks(y, labels)

plt.scatter(x2, y, color='r', label='Median Mid Career Salary')
plt.yticks(y, labels)

plt.xlabel('US Dollars')
plt.title('Starting Median Salary by Major')
plt.legend(loc=2)

degrees = degrees.sort_values(by="mid_p50", ascending=False)
degrees = degrees.reset_index()

fig = plt.figure(figsize=(8, 12))
y = len(degrees.index) - degrees.index
x = degrees['bgn_p50']
x2 = degrees['mid_p50']
labels = degrees['major']

plt.scatter(x, y, color='g', label='Starting Median Salary')
plt.yticks(y, labels)

plt.scatter(x2, y, color='r', label='Median Mid Career Salary')
plt.yticks(y, labels)

plt.xlabel('US Dollars')
plt.title('Salary Information by Major')
plt.legend(loc=2)

degrees = degrees.sort_values(by="mid_p50", ascending=False)
degrees = degrees.reset_index()

fig = plt.figure(figsize=(8, 12))
y = len(degrees.index) - degrees.index
x = degrees['bgn_p50']
x2 = degrees['mid_p50']
x3 = degrees['mid_p25']
x4 = degrees['mid_p75']
x5 = degrees['mid_p10']
x6 = degrees['mid_p90']
labels = degrees['major']

plt.scatter(x5, y, color='r', label='10th pct. Mid Career Salary')
plt.yticks(y, labels)

plt.scatter(x3, y, color='g', label='25th pct. Mid Career Salary')
plt.yticks(y, labels)

plt.scatter(x2, y, color='b', label='Median Mid Career Salary')
plt.yticks(y, labels)

plt.scatter(x4, y, color='#ffff00', label='75th pct. Mid Career Salary')
plt.yticks(y, labels)

plt.scatter(x6, y, color='#00ffff', label='90th pct. Mid Career Salary')
plt.yticks(y, labels)

plt.xlabel('US Dollars')
plt.title('Salary Information by Major')
plt.legend(loc=2)

sal_by_type = sal_by_type.sort_values(by="mid_p50", ascending=False)
sal_by_type = sal_by_type.reset_index()

fig = plt.figure(figsize=(8, 12))
y = len(sal_by_type.index) - sal_by_type.index
x = sal_by_type['bgn_p50']
x2 = sal_by_type['mid_p50']
labels = sal_by_type['school']

plt.scatter(x, y, color='g', label='Starting Median Salary')
plt.yticks(y, labels)

plt.scatter(x2, y, color='r', label='Median Mid Career Salary')
plt.yticks(y, labels)

plt.xlabel('US Dollars')
plt.title('Salary Information by Major')
plt.legend()

plt.show()