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
class Node:
    def __init__(self,feature,threshold,left,right,value) -> None:
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

@dataclass

class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Function to train and construct the decision tree
        
        """
        if (not check_ifreal(X)):
           X= one_hot_encoding(X)
        self.tree= self.Tree(X,y,0)
    def Tree(self,X,y,depth):
        node = Node(None,None,None,None,None)
        if check_ifreal(y):
              if depth >= self.max_depth or len(np.unique(y)) == 1:
                  node.value= np.mean(y)
                  return node
        else:
            if depth >= self.max_depth or len(np.unique(y)) == 1:
                  node.value = np.argmax(np.bincount(y))
                  return node
        feature, splitpoint=  opt_split_attribute(X,y,self.criterion)
        if feature == None:
            if check_ifreal(y):
                  node.value= np.mean(y)
                  return node
            else:
                if (len(y)>0):
                  node.value = y.mode()[0]
                else:
                  node.value =0
                return node
        leftX,rightX,leftY,rightY = split_data(X,y,feature,splitpoint)
        node.feature = feature
        node.threshold = splitpoint
        node.left = self.Tree(leftX,leftY,depth+1)
        node.right = self.Tree(rightX,rightY,depth+1)
        return node     

        # If you wish your code can have cases for different types of input and output data (discrete, real)
        # Use the functions from utils.py to find the optimal attribute to split upon and then construct the tree accordingly.
        # You may(according to your implemetation) need to call functions recursively to construct the tree. 


    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """
        X= one_hot_encoding(X)
        y=[]
        for i in range(X.shape[0]):
            y.append(self._predict_row(self.tree,X.iloc[i]))
        return np.array(y)
        # Traverse the tree you constructed to return the predicted values for the given test inputs.
        
    def _predict_row(self,node,unit) -> float:
         if node.value != None:
             return node.value
         if unit[node.feature] <= node.threshold:
            return self._predict_row(node.left,unit)
         else:
            return self._predict_row(node.right,unit)
    def plot(self) -> None:
        def plot_tree(node, depth=0):
            if node.value is not None:
                print(f"{' ' * depth * 2}Leaf: {node.value}")
            else:
                print(f"{' ' * depth * 2}Node: Feature {node.feature} <= {node.threshold}")
                print(f"{' ' * depth * 2}Left:")
                plot_tree(node.left, depth + 1)
                print(f"{' ' * depth * 2}Right:")
                plot_tree(node.right, depth + 1)

        plot_tree(self.tree)
