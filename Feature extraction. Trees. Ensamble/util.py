import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def read_and_standardize_data(filename, num_of_features):
    
    header_list = []
    for i in range(num_of_features):
        header_list.append('x'+str(i))
    header_list.append('y')
    
    features = header_list[0:num_of_features]
    
    _data = pd.read_csv(filename, header=None, names=header_list) 
    
    X = _data[features]
    y = _data['y']
    
    X = (X.to_numpy())
    y = (y.to_numpy())
#     y = y.reshape(-1, 1)
    
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    
    
    return X, y


def read_and_standardize_mixed_data(filename, num_of_features):
    
    header_list = []
    for i in range(num_of_features):
        header_list.append('x'+str(i))
    header_list.append('y')
    
    features = header_list[0:num_of_features]
    
    _data = pd.read_csv(filename, header=None, names=header_list) 
    
    X = _data[features]
    y = _data['y']
    
    X = (X.to_numpy())
    y = (y.to_numpy())
#     y = y.reshape(-1, 1)
    
    # Pick out continuous variables and standardize them.
    X_continuous = [len(np.unique(X[:, i])) > X.shape[0]*0.05 for i in range(6)]

    X_cont = X[:, X_continuous].copy()

    scaler = StandardScaler().fit(X_cont)
    X_cont = scaler.transform(X_cont)

    X[:, X_continuous] = X_cont.copy()
    
    return X, y


