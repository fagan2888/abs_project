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

def logistic_regression(penalty, data, y, predictors, alpha, models_to_plot={}):
    #Fit the model
    logreg = LogisticRegression(penalty = penalty, C=alpha)
    logreg.fit(data[predictors],data[y])
    y_pred = logreg.predict(data[predictors])
    
    #Check if a plot is to be made for the entered alpha
    if alpha in models_to_plot:
        plt.subplot(models_to_plot[alpha])
        plt.tight_layout()
        plt.plot(data['original_combined_ltv'],y_pred,'ro')
        plt.plot(data['original_combined_ltv'],data[y],'.')
        plt.title('Plot for alpha: %.3g'%alpha)
    
    #Return the result in pre-defined format
    rss = sum((y_pred-data[y])**2)
    ret = [rss]
    ret.extend(np.append(logreg.intercept_,logreg.coef_))
    #ret.extend(logreg.coef_)
    return ret    

def ridge_regression_optimizer(alpha, data, y, predictors):
    #Fit the model
    ridgereg = Ridge(alpha=alpha,normalize=True)
    ridgereg.fit(data[predictors],data[y])
    y_pred = ridgereg.predict(data[predictors])
    
    #Return the result in pre-defined format
    rss = sum((y_pred-data[y])**2)
    return rss

### Ridge regression
def ridge_regression(data, y, predictors, alpha):
    #Fit the model
    ridgereg = Ridge(alpha=alpha,normalize=True)
    ridgereg.fit(data[predictors],data[y])
    y_pred = ridgereg.predict(data[predictors])
    
    #Check if a plot is to be made for the entered alpha
    plt.plot(data['original_combined_ltv'],y_pred,'ro')
    plt.plot(data['original_combined_ltv'],data[y],'.')
    plt.title('Plot for alpha: %.3g'%alpha)
    #plt.show()
    
    #Return the result in pre-defined format
    rss = sum((y_pred-data[y])**2)
    ret = [rss]
    ret.extend([ridgereg.intercept_])
    ret.extend(ridgereg.coef_)
    return ret

### Lasso Regression

def lasso_optimizer(alpha, data, y, predictors):
    #Fit the model
    lassoreg = Lasso(alpha=alpha,normalize=True, max_iter=1e5)
    lassoreg.fit(data[predictors],data[y])
    y_pred = lassoreg.predict(data[predictors])
    
    #Return the result in pre-defined format
    rss = sum((y_pred-data[y])**2)
    return rss

def lasso_regression(data, y, predictors, alpha):
    #Fit the model
    lassoreg = Lasso(alpha=alpha,normalize=True, max_iter=1e5)
    lassoreg.fit(data[predictors],data[y])
    y_pred = lassoreg.predict(data[predictors])
    
    #Check if a plot is to be made for the entered alpha
    plt.plot(data['original_combined_ltv'],y_pred,'ro')
    plt.plot(data['original_combined_ltv'],data[y],'.')
    plt.title('Plot for alpha: %.3g'%alpha)
    #plt.show()
    
    #Return the result in pre-defined format
    rss = sum((y_pred-data[y])**2)
    ret = [rss]
    ret.extend([lassoreg.intercept_])
    ret.extend(lassoreg.coef_)
    return ret
