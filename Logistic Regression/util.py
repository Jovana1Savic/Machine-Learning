# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 19:37:58 2020

@author: Jovana
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def read_standardize_split_data(file_path):
    
    header_list = ['x1', 'x2', 'x3', 'x4', 'x5', 'y']
    features = header_list[0:5]
    
    _data = pd.read_csv(file_path, header=None, names=header_list) 
    #print(_data.describe())
    
    X = _data[features]
    y = _data['y']
    
    X = (X.to_numpy())
    y = (y.to_numpy()).T
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=2)
    
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test