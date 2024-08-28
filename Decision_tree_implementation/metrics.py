from typing import Union
import pandas as pd
import numpy as np


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    print(y_hat.size, y.size)
    assert y_hat.size == y.size
    # TODO: Write here
    
    correct_predictions = (y_hat == y).sum()
    accuracy = correct_predictions / y.size
    
    return accuracy *100


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    fp_tp=0
    tp=0
    for i in range(len(y)):
        if (y_hat[i]==cls):
            fp_tp+=1
            if (y[i]==cls):
                tp+=1
    return tp/fp_tp
        


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    fp_tp=0
    tp=0
    for i in range(len(y)):
        if (y[i]==cls):
            fp_tp+=1
            if (y_hat[i]==cls):
                tp+=1
    return tp/fp_tp


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    mse = np.sqrt(np.sum((y-y_hat)**2))
    return mse


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    
    """
    return np.sum(abs(y-y_hat))
