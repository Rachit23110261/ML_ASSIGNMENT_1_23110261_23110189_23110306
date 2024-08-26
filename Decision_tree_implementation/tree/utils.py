"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
import numpy as np

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """
    for column in X.columns:
        unique_values = sorted(X[column].unique())
        value_to_index = {value: index for index, value in enumerate(unique_values)}
        X[column] = X[column].apply(lambda x: value_to_index[x])
    
    return X

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    unique= np.unique(y)
    if ( len(unique)< len(y)/2):
        return False
    else:
        return True


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    unique = np.unique(Y)
    entropy=0
    for i in unique:
        p=0
        for j in Y :
            if j==i:
                p+=1/len(Y)
        if p not in [0,1]:
          entropy -= p*np.log2(p)+(1-p)*np.log2(1-p)
    return entropy

def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    unique = np.unique(Y)
    gini=1
    for i in unique:
        p=0
        for j in Y :
            if j==i:
                p+=1/len(Y)
        gini -=  p**2
    return gini
        
def mse(Y: pd.Series) -> float:
    """
    Function to calculate the Mean Squared Error (MSE) of a pandas Series Y.
    """
    mean_Y = np.mean(Y)
    return np.mean((Y - mean_Y) ** 2)    

def opt_value(X: pd.Series, Y: pd.Series, criterion):
    X= np.sort(X)
    mid =[]
    max_gain=0
    max_value=None
    for i in range(len(X)-1):
        mid.append((X[i]+X[i+1])/2)
    for j in mid:
        X_left = X[X<j]
        X_right= X[X>=j]
        Y_left = X[X<j]
        Y_right= X[X>=j]
        gain = information_gain(Y_left,X_left,criterion) + information_gain(Y_right,X_right,criterion)
        max_gain = max(gain, max_gain )
        if max_gain==gain:
            max_value= j
    return max_value
        
def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)
    
    """
    n= len(Y)
    if n!=0:
        if (criterion=='entropy'):
            print(Y.shape, attr.shape)
            df_entropy = pd.DataFrame({'Y': Y , 'attr': attr})
            df_group =  df_entropy.groupby('attr')
            
            gain = entropy(Y)
            for name , group in df_group:
                gain -= (group.shape[0]/n)*entropy(group['Y'])
            return gain
        elif (criterion=='gini index'):
            df_entropy = pd.DataFrame({'Y': Y , 'attr': attr})
            df_group =  df_entropy.groupby('attr')
            
            gain = gini_index(Y)
            for name , group in df_group:
                gain -= (group.shape[0]/n)*gini_index(group['Y'])
            return gain
        else:
            df_entropy = pd.DataFrame({'Y': Y , 'attr': attr})
            df_group =  df_entropy.groupby('attr')
            gain = mse(Y)
            for name , group in df_group:
                gain -= (group.shape[0]/n)*mse(group['Y'])
            return gain
    else:
        return 0      
            
            

def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """
    best_gain = -np.inf 
    best_attr = None
    for feature in features:
        attr = X[feature]
        if criterion=='information_gain':
            if check_ifreal(y):
                gain = information_gain(y,attr, 'mse')
                if gain > best_gain: 
                        best_gain = gain
                        best_attr = feature
            else:
                gain = information_gain(y,attr, 'entropy')
                if gain > best_gain: 
                        best_gain = gain
                        best_attr = feature
                
        elif criterion=='gini_index':
            gain = information_gain(y,attr, 'gini index')
            if gain > best_gain:  
                        best_gain = gain
                        best_attr = feature
    return best_attr
    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).


def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """
    if check_ifreal(X[attribute]):
        x_1= X[X[attribute]>value]
        x_2= X[X[attribute]<=value]
    else:
        x_1= X[X[attribute]==value]
        x_2= X[X[attribute]!=value]
    if check_ifreal(y):
        y_1= y[X[attribute]>value]
        y_2= y[X[attribute]<=value]
    else:
        y_1= y[X[attribute]==value]
        y_2= y[X[attribute]!=value]
    return x_1,x_2,y_1,y_2
        
    
    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.

# x = pd.DataFrame({"A":[1,2,3],"B":[1,1,0]})
# y = pd.Series([0,1,0])
# print(opt_split_attribute(x,y,'information gain',["A","B"]))
# print(split_data(x,y,'A',1))