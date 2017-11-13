#! /usr/bin/python3

#data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# Acquire data
train_df = pd.read_csv('datasets/titanic/train.csv')
test_df = pd.read_csv('datasets/titanic/test.csv')
combine = [train_df, test_df]

# Analyze by describing data
# print(train_df.columns.values)
# print(train_df.head())

# print(train_df.info())
# print('_'*40)
# print(test_df.info())

# print(train_df.describe())
# print(train_df.describe(include=['O']))

# Analyze by pivoting features
# print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
# print(train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
# print(train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))

# Analyze by visualizing data
# g = sns.FacetGrid(train_df, col='Survived')
# g.map(plt.hist, 'Age', bins=20)

# grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
# grid.map(plt.hist, 'Age', alpha=.5, bins=20)
# grid.add_legend();

# grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
# grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep', order=[1,2,3], hue_order=["female","male"])
# grid.add_legend()

# grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
# grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
# grid.add_legend()

# Wrangle data
# print('Before', train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]
# print("After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

# print(pd.crosstab(train_df['Title'], train_df['Sex']))

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# print(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

title_mapping = {
    'Mr': 1,
    'Miss': 2,
    'Mrs': 3,
    'Master': 4,
    'Rare': 5
}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]

for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

# grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
# grid.map(plt.hist, 'Age', alpha=.5, bins=20)
# grid.add_legend()

guess_ages = np.zeros((2, 3))
for dataset in combine:
    for i in range(2):
        for j in range(3):
            guess_df = dataset[(dataset['Sex'] == 1) & (dataset['Pclass'] == j+1)]['Age'].dropna()

            age_guess = guess_df.median()
            guess_ages[i, j] = int(age_guess/0.5 + 0.5) * 0.5
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[
                (dataset.Age.isnull()) &
                (dataset.Sex == i) &
                (dataset.Pclass == j+1),
                'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
# print(train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True))

for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']

train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]

for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

# print(train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False))

for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

# print(train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())

train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

# print(train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10))

freq_port = train_df.Embarked.dropna().mode()[0] # most frequent used port
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)

# print(train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True))

for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]

# print(train_df.head(10))
# print(test_df.head(10))

# Model
X_train = train_df.drop('Survived', axis=1)
Y_train = train_df['Survived']
X_test = test_df.drop('PassengerId', axis=1).copy()

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
# Random Forrest
data = trainData(X_train, Y_train, 'random_forest')
acc_data = accuracy(data, X_train, Y_train)
score.append(acc_data)
Y_pred = predict(data, X_test)
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
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('datasets/titanic/submission.csv', index=False)