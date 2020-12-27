import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
from sklearn.model_selection import KFold


def plot_cross_validation_results(X, y, model, param_name, param_range, scoring, folds, 
                            fig_test, fig_train, label, color, color_train=None, std=False):
    """
    Perform cross-validation for given model for given hyper-parameter. 
    Plot average train and test accuracy on given figures.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples, )
    param_range : array-like of shape (n_params,)
    param_name : str
    scoring : str
    folds : int
        number of folds to use in cross validation
    fig_test : int or str
        figure in which test results will be plotted
    fig_train : int or str
        figure in which train results will be plotted
    label : str
        plot label
    color : str
        plot color for test samples
    color_train : str (default None)
        plot color for train samples, if None same as color
    std : bool (default False)
        plot standard deviation as well
    
    """
    
    train_scores_sk, test_scores_sk = validation_curve(
                        model, X, y,
                        param_name=param_name,
                        param_range=param_range,
                        cv=KFold(n_splits=folds, shuffle=True, random_state=42),
                        scoring=scoring)
    
    train_scores_mean = np.mean(train_scores_sk, axis=1)
    train_scores_std = np.std(train_scores_sk, axis=1)
    test_scores_mean = np.mean(test_scores_sk, axis=1)
    test_scores_std = np.std(test_scores_sk, axis=1)
    
    if color_train is None:
        color_train = color_test
    
    # Parameters are multidimensional, use range.
    if (np.array(param_range)).ndim > 1:
        param_range = range(len(param_range))
        
        
    plt.figure(fig_train)
    
    if label == "":
        label = "trening"
    plt.plot(param_range, train_scores_mean, label=label, color=color_train)
    
    if std:
        plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color=color_train)
    
    plt.legend(loc="best")
    
    plt.figure(fig_test)
    if label == "trening":
        label = "test"
    plt.plot(param_range, test_scores_mean, label=label, color=color)
    
    if std:
        plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color=color)
        
    plt.legend(loc="best")
    
    return test_scores_mean, test_scores_std, train_scores_mean, train_scores_std


import matplotlib.cm as cm
from sklearn.base import clone

def plot_cross_validation_multiparam(X, y, model, main_param_name, main_param_range, 
                                     param_name, param_range,
                                     scoring, fig_id, folds=5, 
                                     colors=None, std=False):
    
    if colors is None:
        colors = cm.rainbow(np.linspace(0, 1, len(param_range)))
  
    fig_train = fig_id + str(0)
    fig_test = fig_id + str(1)
    
    plt.figure(num = fig_train, figsize=(10, 7))
    plt.suptitle("Rezultati na obučavajućem skupu", fontsize=20)
    plt.xlabel("Veličina ansambla", fontsize=16)
    plt.ylabel("Skor", fontsize=16)
    
    plt.figure(num = fig_test, figsize=(10, 7))
    plt.suptitle("Rezultati na validacionom skupu", fontsize=20)
    plt.xlabel("Veličina ansambla", fontsize=16)
    plt.ylabel("Skor", fontsize=16)
    
    for (i,param) in enumerate(param_range):
        m = clone(model)
        m.set_params(**{param_name: param})
        plot_cross_validation_results(X, y, model=m, param_name=main_param_name, param_range=main_param_range,
                                      scoring=scoring, folds=folds, 
                                      fig_test=fig_test, fig_train=fig_train,
                                      label=param_name + " = " + str(param),
                                      color = colors[i], color_train=colors[i], std=std)
        
#     plt.figure(fig_train)
#     plt.show()
    
#     plt.figure(fig_test)
#     plt.show()
    
    return