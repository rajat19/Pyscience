import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

train_df = pd.read_csv('datasets/house/train.csv')
test_df = pd.read_csv('datasets/house/test.csv')
combine = [train_df, test_df]

#correlation matrix
corrmat = train_df.corr()
# f, ax = plt.subplots(figsize=(12, 9))
# sns.heatmap(corrmat, vmax=.8, square=True);

k = 20 #number of variables for heatmap
# cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
# cm = np.corrcoef(train_df[cols].values.T)
# sns.set(font_scale=1.25)
# hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

"""
1. Deviate from normal distribution
2. Have appreciable positive skewness
3. Show peakedness (check by kurtosis)
"""
# sns.distplot(train_df['SalePrice'])
# print("Skewness: %f" % train_df['SalePrice'].skew())
# print("Kurtosis: %f" % train_df['SalePrice'].kurt())

"""
MSSubClass and SalePrice
"""
# print(train_df[['MSSubClass', 'SalePrice']].groupby(['MSSubClass'], as_index=False).mean().sort_values(by='SalePrice', ascending=False))

"""
MSZoning and SalePrice
1. Floating Village Residential on an average has highest selling price
"""
# print(train_df[['MSZoning', 'SalePrice']].groupby(['MSZoning'], as_index=False).mean().sort_values(by='SalePrice', ascending=False))

"""
LotArea and SalePrice
"""

"""
Top Features Affecting Sale Price
1. OverallQual .79
2. GrLivArea .71
3. GarageCars .64
4. TotalBsmtSF .61
5. 1stFlrSF .61
6. FullBath .56
7. YearBuilt .52
8. Fireplaces .47
9. WoodDeckSF .32
10. 2ndFlrSF .32
11. OpenPorchSF .32
"""

train_filtered = train_df[[
    'SalePrice',
    'OverallQual',
    'GrLivArea',
    'GarageCars',
    'TotalBsmtSF',
    'YearBuilt',
]]
test_filtered = test_df[[
    'Id',
    'OverallQual',
    'GrLivArea',
    'GarageCars',
    'TotalBsmtSF',
    'YearBuilt',
]]

test_filtered['GarageCars'] = test_filtered['GarageCars'].fillna(0).astype(int)
test_filtered['TotalBsmtSF'] = test_filtered['GarageCars'].fillna(0).astype(int)

# print(test_filtered.info())
# print(train_filtered.sort_values(by='SalePrice', ascending=False).head());

X_train = train_filtered.drop('SalePrice', axis=1)
Y_train = train_filtered['SalePrice']
X_test = test_filtered.drop('Id', axis=1).copy()

"""
Training is done by following methods
Logistic Regression
KNN or k-Nearest Neighbors
Support Vector Machines
Naive Bayes classifier
Decision Tree
Random Forrest
Perceptron
Artificial neural network
RVM or Relevance Vector Machine
"""
def trainData(X_train, Y_train, type):
    fn = ''
    if type == 'logreg':
       fn = LogisticRegression() 
    elif type == 'svc':
        fn = SVC()
    elif type == 'knn':
        fn = KNeighborsClassifier()
    elif type == 'gaussian':
        fn = GaussianNB()
    elif type == 'perceptron':
        fn = Perceptron()
    elif type == 'linear_svc':
        fn = LinearSVC()
    elif type == 'sgd':
        fn = SGDClassifier()
    elif type == 'decision_tree':
        fn = DecisionTreeClassifier()
    elif type == 'random_forest':
        fn = RandomForestClassifier()
    return fn.fit(X_train, Y_train)

def predict(data, X_test):
    return data.predict(X_test)

def accuracy(data, X_train, Y_train):
    return round(data.score(X_train, Y_train) * 100 , 2)

score = []
# Logistic Regression
data = trainData(X_train, Y_train, 'logreg')
acc_data = accuracy(data, X_train, Y_train)
score.append(acc_data)
# KNN or k-Nearest Neighbors
data = trainData(X_train, Y_train, 'knn')
acc_data = accuracy(data, X_train, Y_train)
score.append(acc_data)
# Support Vector Machines
data = trainData(X_train, Y_train, 'svc')
acc_data = accuracy(data, X_train, Y_train)
score.append(acc_data)
# Naive Bayes classifier
data = trainData(X_train, Y_train, 'gaussian')
acc_data = accuracy(data, X_train, Y_train)
score.append(acc_data)
# Decision Tree
data = trainData(X_train, Y_train, 'decision_tree')
acc_data = accuracy(data, X_train, Y_train)
score.append(acc_data)
Y_pred = predict(data, X_test)
# Random Forrest
data = trainData(X_train, Y_train, 'random_forest')
acc_data = accuracy(data, X_train, Y_train)
score.append(acc_data)
# Perceptron
data = trainData(X_train, Y_train, 'perceptron')
acc_data = accuracy(data, X_train, Y_train)
score.append(acc_data)
# Linear SVC
data = trainData(X_train, Y_train, 'linear_svc')
acc_data = accuracy(data, X_train, Y_train)
score.append(acc_data)
# Stochastic Gradient Descent
data = trainData(X_train, Y_train, 'sgd')
acc_data = accuracy(data, X_train, Y_train)
score.append(acc_data)
# coeff_df = pd.DataFrame(train_df.columns.delete(0))
# coeff_df.columns = ['Feature']
# coeff_df["Correlation"] = pd.Series(data.coef_[0])
# print(coeff_df.sort_values(by='Correlation', ascending=False))

models = pd.DataFrame({
    'Model': [
        'Logistic Regression',
        'KNN',
        'Support Vector Machines',
        'Naive Bayes',
        'Decision Tree',
        'Random Forest',
        'Perceptron',
        'Linear SVC',
        'Stochastic Gradient Decent',
    ],
    'Score': score
})
# print(models)
# print(models.sort_values(by='Score', ascending=False))
submission = pd.DataFrame({
        "Id": test_filtered['Id'],
        "SalePrice": Y_pred
    })
submission.to_csv('datasets/house/submission.csv', index=False)

plt.show()