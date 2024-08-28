import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification

np.random.seed(42)

# Code given in the question
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
plt.scatter(X[:, 0], X[:, 1], c=y)
X=pd.DataFrame(X)
y=pd.Series(y)
X_train= X[:X.shape[0]*7//10]
y_train= y[:y.shape[0]*7//10]
X_test= X[X.shape[0]*7//10:]
y_test= y[y.shape[0]*7//10:]
for criteria in ["information_gain", "gini_index"]:
    tree = DecisionTree(criterion=criteria)  # Split based on Inf. Gain
    tree.fit(X_train, y_train)
    y_hat = tree.predict(X_test)
    tree.plot()
    print("Criteria :", criteria)
    print("RMSE: ", rmse(y_hat, y_test))
    print("MAE: ", mae(y_hat, y_test))
    print("Criteria :", criteria)
    print("Accuracy: ", accuracy(y_hat, y_test))
k = 5
predictions = {}
accuracies = []
fold_size = len(X) // k
for i in range(k):
    test_start = i * fold_size
    test_end = (i + 1) * fold_size
    test_set = X[test_start:test_end]
    test_labels = y[test_start:test_end]
    training_set = np.concatenate((X[:test_start], X[test_end:]), axis=0)
    training_labels = np.concatenate((y[:test_start], y[test_end:]), axis=0)
    training_set = pd.DataFrame(training_set)
    training_labels = pd.DataFrame(training_labels)
    dt_classifier = DecisionTree(criterion=criteria) 
    dt_classifier.fit(training_set, training_labels)
    fold_predictions = dt_classifier.predict(test_set)
    fold_accuracy = np.mean(fold_predictions == test_labels)
    predictions[i] = fold_predictions
    accuracies.append(fold_accuracy)
from itertools import product
for i in range(k):
    print("Fold {}: Accuracy: {:.4f}".format(i+1, accuracies[i]))    
    
# nested cross-validation
outer_folds = 5
inner_folds = 3
fold_size = len(X) // outer_folds
outer_accuracies = []
best_inner_accuracies = []
for outer_i in range(outer_folds):
    test_start = outer_i * fold_size
    test_end = (outer_i + 1) * fold_size if outer_i < outer_folds - 1 else len(X)
    test_indices = range(test_start, test_end)
    train_indices = list(range(0, test_start)) + list(range(test_end, len(X)))
    X_train_outer = X.iloc[train_indices]
    y_train_outer = y.iloc[train_indices]
    X_test_outer = X.iloc[test_indices]
    y_test_outer = y.iloc[test_indices]
    best_accuracy = 0
    best_criterion = None
    for criterion in ["information_gain", "gini_index"]:
        inner_accuracies = []
        for inner_i in range(inner_folds):
            # Split data for the current inner fold
            inner_test_start = inner_i * (len(X_train_outer) // inner_folds)
            inner_test_end = (inner_i + 1) * (len(X_train_outer) // inner_folds) if inner_i < inner_folds - 1 else len(X_train_outer)
            inner_test_indices = range(inner_test_start, inner_test_end)
            inner_train_indices = list(range(0, inner_test_start)) + list(range(inner_test_end, len(X_train_outer)))
            X_train_inner = X_train_outer.iloc[inner_train_indices]
            y_train_inner = y_train_outer.iloc[inner_train_indices]
            X_test_inner = X_train_outer.iloc[inner_test_indices]
            y_test_inner = y_train_outer.iloc[inner_test_indices]
            tree = DecisionTree(criterion=criterion)
            tree.fit(X_train_inner, y_train_inner)
            y_hat_inner = tree.predict(X_test_inner)
            fold_accuracy = accuracy(y_hat_inner, y_test_inner)
            inner_accuracies.append(fold_accuracy)
        avg_accuracy = np.mean(inner_accuracies)
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_criterion = criterion
    tree = DecisionTree(criterion=best_criterion)
    tree.fit(X_train_outer, y_train_outer)
    y_hat_outer = tree.predict(X_test_outer)
    outer_accuracy = accuracy(y_hat_outer, y_test_outer)
    outer_accuracies.append(outer_accuracy)
    best_inner_accuracies.append(best_accuracy)
    print(f"Outer Fold {outer_i + 1}:")
    print(f"Best Inner Accuracy: {best_accuracy:.4f}")
    print(f"Outer Accuracy with Best Criterion ({best_criterion}): {outer_accuracy:.4f}")
    tree.plot()
    print("RMSE: ", rmse(y_hat_outer, y_test_outer))
    print("MAE: ", mae(y_hat_outer, y_test_outer))
    print("Accuracy: ", accuracy(y_hat_outer, y_test_outer))
print(f"Average Outer Accuracy: {np.mean(outer_accuracies):.4f}")
print(f"Average Best Inner Accuracy: {np.mean(best_inner_accuracies):.4f}")

