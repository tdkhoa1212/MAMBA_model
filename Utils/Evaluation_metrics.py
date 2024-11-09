import numpy as np
from scipy.stats import pearsonr, spearmanr

def information_coefficient(y_true, y_pred):
    return pearsonr(y_pred, y_true)[0]

def rank_information_coefficient(y_true, y_pred):
    return spearmanr(y_pred, y_true)[0]

def RMSE(y_true, y_pred):
    return np.sqrt(((y_pred - y_true) ** 2).mean())
