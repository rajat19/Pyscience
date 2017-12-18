import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.offline as py
import seaborn as sns

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('datasets/zoo/class.csv')
zoo = pd.read_csv('datasets/zoo/zoo.csv')

zoo = zoo.drop(['animal_name'], axis=1)

# Model
X_train = zoo.drop('class_type', axis=1)
Y_train = zoo['class_type']

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
        fn = Perceptron(tol=None, max_iter=5)
    elif type == 'linear_svc':
        fn = LinearSVC()
    elif type == 'sgd':
        fn = SGDClassifier(tol=None, max_iter=5)
    elif type == 'decision_tree':
        fn = DecisionTreeClassifier()
    elif type == 'random_forest':
        fn = RandomForestClassifier()
    return fn.fit(X_train, Y_train)

def predict(data, X_test):
    return data.predict(X_test)

def accuracy(data, X_train, Y_train):
    # return round(data.score(X_train, Y_train) * 100 , 2)
    return data.score(X_train, Y_train)

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
print(models.sort_values(by="Score", ascending=False))

# Performance of models
fg,ax = plt.subplots(nrows=1,ncols=1,figsize=(6, 2.5))
sns.barplot(models['Model'], score, palette="RdBu")
plt.ylim(0.85, 1)
ax.set_ylabel("Performance")
ax.set_xlabel("Name")
ax.set_xticklabels(models['Model'],rotation=35)
plt.title('Battle of Algorithms')

plt.show()