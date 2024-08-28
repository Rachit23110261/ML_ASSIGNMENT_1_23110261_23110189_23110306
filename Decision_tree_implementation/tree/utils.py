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
    df = X.copy()  # Avoid modifying the original DataFrame
    for col in df.columns:
        if not check_ifreal(df[col]):
            df = pd.get_dummies(df, columns=[col], drop_first=True)  # drop_first=True to avoid multicollinearity
    return df*1
    
    return X*1 # for converting to integer from bool

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    if pd.api.types.is_string_dtype(y):
        return False
    if pd.api.types.is_numeric_dtype(y):
        if pd.api.types.is_float_dtype(y):
            return True
        else:
            return False
    return False

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
          entropy -= p*np.log2(p) + (1-p)*np.log2(1-p)
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

        
def information_gain(Y: pd.Series, Y_left, Y_right,criteria ) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)
    
    """
    if criteria=='information_gain':
        if check_ifreal(Y):
            return mse(Y) - (Y_left.size / Y.size) * mse(Y_left) - (Y_right.size / Y.size) * mse(Y_right)
        
        else:
            weightLeft, weightRight = Y_left.size / Y.size, Y_right.size / Y.size
            return entropy(Y) - (weightLeft * entropy(Y_left) + weightRight * entropy(Y_right))   
    else:
        weightLeft, weightRight = Y_left.size / Y.size, Y_right.size / Y.size
        return gini_index(Y) - (weightLeft * gini_index(Y_left) + weightRight * gini_index(Y_right))
            
            

def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """
    best_gain = -np.inf 
    best_attr = None
    best_split =None

    for feature in X.columns:
        attr = X[feature]
        unique_attr= attr.unique()
        possible_splits= (unique_attr[1:]+ unique_attr[:-1])/2
        for split in possible_splits:
            X_left,X_right,y_left,y_right = split_data(X,y,feature,split)
            if criterion == "information_gain":
                currentScore = information_gain(y, y_left, y_right,'information_gain')
            else:
                currentScore = information_gain(y, y_left, y_right,'giniindex')

            if currentScore > best_gain:
                best_gain = currentScore
                best_attr = feature
                best_split = split
    return best_attr, best_split
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
    df=X.copy()
    df['Y']=y
    left = df[df[attribute] <= value]
    right = df[df[attribute] > value]
    y_left, y_right = left["Y"], right["Y"]
    X_left, X_right = left.drop(columns=["Y"]), right.drop(columns=["Y"])
    return X_left,X_right,y_left, y_right
    
    
        
    
    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.

# x = pd.DataFrame({"A":[1,2,3],"B":[1,1,0]})
# y = pd.Series([0,1,0])
# print(opt_split_attribute(x,y,'information gain',["A","B"]))
# print(split_data(x,y,'A',1))