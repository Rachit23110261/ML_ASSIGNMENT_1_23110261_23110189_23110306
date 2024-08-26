"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *

np.random.seed(42)


@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X: pd.DataFrame, y: pd.Series, depth: int = 0):
        """
        Function to train and construct the decision tree
        
        """
        
        
        print(X.shape,y.shape)
        if check_ifreal(y):
              if depth >= self.max_depth or len(np.unique(y)) == 1:
                  return np.mean(y)
        else:
            
            if depth >= self.max_depth or len(np.unique(y)) == 1:
                  return np.argmax(np.bincount(y))
            # X = one_hot_encoding(X)
            # y = np.array(one_hot_encoding(pd.DataFrame(y)))
        X=pd.DataFrame(X)
        print(X.shape,y.shape)
        attribute = opt_split_attribute(X, y, self.criterion, X.columns)
        value = None
        subtrees=[]
        if check_ifreal(X[attribute]):
            value = opt_value(X[attribute],y,self.criterion)
            x_1, x_2, y_1, y_2 = split_data(X, y, attribute, value)
            subtrees.append( self.fit(x_1, y_1, depth + 1))
            subtrees.append(self.fit(x_2, y_2, depth + 1))
        else:
            for val in np.unique(X[attribute]):
                x_sub = X[X[attribute] == val]
                y_sub = y[X[attribute] == val]
                subtree = self.fit(x_sub, y_sub, depth + 1)
                subtrees.append( subtree)
        self.tree = {
                'attribute': attribute,
                'value': value,
                'subtrees': subtrees
        }

        return self.tree

        # If you wish your code can have cases for different types of input and output data (discrete, real)
        # Use the functions from utils.py to find the optimal attribute to split upon and then construct the tree accordingly.
        # You may(according to your implemetation) need to call functions recursively to construct the tree. 


    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """
        isreal =check_ifreal(X.iloc[:,0])
        # Traverse the tree you constructed to return the predicted values for the given test inputs.
        return np.array([self._predict_row(row, self.tree,isreal) for _, row in X.iterrows()])
    def _predict_row(self, row: pd.Series, tree: dict, isreal) -> float:
        if isinstance(tree, dict):
            if isreal:
                if row[tree['attribute']] > tree['value']:
                    return self._predict_row(row, tree['subtrees'][0],isreal)
                else:
                    return self._predict_row(row, tree['subtrees'][1],isreal)
            else:
                if row[tree['attribute']] == tree['value']:
                    return self._predict_row(row, tree['left'])
                else:
                    return self._predict_row(row, tree['right'])
        else:
            return tree
    def plot(self) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        pass
