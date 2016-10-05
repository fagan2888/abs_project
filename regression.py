# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 19:00:36 2016

@author: Chandler
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression

def variable_dist(loan_df, target_variable, models_to_plot):
    for i in range(len(models_to_plot)):
        if i == 9:
            plt.subplot(2,5,10)
            loan_df[target_variable[9]].hist()
            plt.title('Histogram of %s'%target_variable[9])
            plt.xlabel(target_variable[9])
            plt.ylabel('Frequency')
        else:
            plt.subplot(models_to_plot[target_variable[i]])
            loan_df[target_variable[i]].hist()
            plt.title('Histogram of %s'%target_variable[i])
            plt.xlabel(target_variable[i])
            plt.ylabel('Frequency')
    return loan_df


### Logistic regression ###

def logistic_regression(penalty, data, y, predictors, alpha):
    #Fit the model
    logreg = LogisticRegression(penalty = penalty, C=alpha[0])
    logreg.fit(data[predictors],data[y])
    y_pred = logreg.predict(data[predictors])
    
    plt.plot(data['original_combined_ltv'],y_pred,'ro')
    plt.plot(data['original_combined_ltv'],data[y],'.')
    plt.title('Plot for alpha: %.3g'%alpha[0])
    #plt.show()
    
    #Return the result in pre-defined format
    rss = sum((y_pred-data[y])**2)
    ret = [rss]
    ret.extend(np.append(logreg.intercept_,logreg.coef_))
    #ret.extend(logreg.coef_)
    return ret    

def RL_optimizer(alpha, data, y, predictors, func):
    # func = Ridge or Lasso
    #Fit the model
    ridgereg = func(alpha=alpha,normalize=True)
    ridgereg.fit(data[predictors],data[y])
    y_pred = ridgereg.predict(data[predictors])
    
    #Return the result in pre-defined format
    rss = sum((y_pred-data[y])**2)
    return rss

### Ridge regression
def RL_regression(data, y, predictors, alpha, func):
    # func = Ridge or Lasso
    #Fit the model

    rl_reg = func(alpha=alpha,normalize=True)
    rl_reg.fit(data[predictors],data[y])
    y_pred = rl_reg.predict(data[predictors])

    #Check if a plot is to be made for the entered alpha
    plt.plot(data['original_combined_ltv'],y_pred,'ro')
    plt.plot(data['original_combined_ltv'],data[y],'.')
    plt.title('Plot for alpha: %.3g'%alpha)
    #plt.show()
    
    #Return the result in pre-defined format
    rss = sum((y_pred-data[y])**2)

    #ret = [rss]
    #ret.extend([rl_reg.intercept_])
    #ret.extend(rl_reg.coef_)

    ret = rl_reg.coef_

    return ret, rss
