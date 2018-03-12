# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 12:34:27 2018

@author: gring
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import neighbors

# Seed setting for reproducibility purposes
np.random.seed(seed = 1)

# Change background of sns plots
sns.set(color_codes = True)

# Origin of data
ws = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Loading of Iris data
df_iris = pd.read_csv(ws)

# Addition of column names
df_iris.columns = ["sepal.length",
                   "sepal.width",
                   "petal.length",
                   "petal.width",
                   "class"]

# Bivariate plot with Petal Length against Sepal Length with classification of
# plants as fill in plot
sns.lmplot(x = "petal.length",
           y = "sepal.length",
           hue = "class",
           fit_reg = False,
           data = df_iris)

# Training data subsetting
n = len(df_iris)
train_size = 0.8 # Training set % of total dataset

# Training set length
train_subset = int(n*train_size)

# Random selection for training set
rand_sel = np.random.choice(n,
                            train_subset,
                            replace = False)

# Subsetting of data
train_idx = df_iris.index.isin(rand_sel)

# Creation of training and testing sets
df_iris_train = df_iris[train_idx]
df_iris_test = df_iris[~train_idx]

# sepal.length and petal.length will act as features for classification
# target will be the plant class

X = df_iris_train[["sepal.length", "petal.length"]]
y = df_iris_train["class"]

# Number of neighbours considered for the K-NN algorithm to be compared
n_neighbours = np.arange(1, 100)
# Array of zeros to store accuracies to be stored
Accuracy = np.zeros(len(n_neighbours))

for i in n_neighbours:
  # Initialization of K-NN classifier model
  knn_mod = neighbors.KNeighborsClassifier(i,
                                         weights = "distance")

  # Fitting of K-NN model
  knn_mod.fit(X, y)

  # Prediction of targets using variables used to fit model from test dataset
  X_pred = df_iris_test[["sepal.length", "petal.length"]]
  y_pred = knn_mod.predict(X_pred)

  # Test targets
  y_test = df_iris_test["class"]
  # Accuracy calculation
  Accuracy[i-1] = np.round(sum(y_pred == y_test) / len(y_pred), 6)

# Column stacking of the two numpy arrays
accuracy = np.column_stack((n_neighbours, Accuracy))
# Dataframe with column stacked results
df_accuracy = pd.DataFrame(data = accuracy, columns = ["k", "Accuracy"])

# Plot of k vs accuracy to select the optimum k
sns.lmplot(x = "k",
           y = "Accuracy",
           fit_reg = False,
           data = df_accuracy)