# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 15:37:58 2016

@author: Chandler
"""
import os
os.chdir('C:/Users/Chandler/Desktop/Berkeley/2016 Fall/230M ABS/abs_project')
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn import tree
import regression as r

loan_df = pd.read_csv('C:/Users/Chandler/Desktop/Berkeley/2016 Fall/230M ABS/abs_project/freddie_final.csv')

### show histogram of different variable
target_variable = ['credit_score','first_time_homebuyer_flag','original_combined_ltv','original_upb','original_ltv','original_interest_rate','prepayment_penalty_flag','number_of_borrowers','Prepay Percent','Home Price']
models_to_plot = {'credit_score':251,'first_time_homebuyer_flag':252,'original_combined_ltv':253,'original_upb':254,'original_ltv':255,'original_interest_rate':256,'prepayment_penalty_flag':257,'number_of_borrowers':258,'Prepay Percent':259,'Home Price':2510}

# r.variable_dist(loan_df, target_variable, models_to_plot)

### Filter out the prepay percent only with <80%
loan_df = loan_df[loan_df['Prepay Percent']<0.8]
###

###
loan_df['Prepayment'] = 1 if loan_df['Prepay Percent']
###

### ridge regression
predictors=['credit_score','first_time_homebuyer_flag','original_combined_ltv','original_upb','original_ltv','original_interest_rate','prepayment_penalty_flag','number_of_borrowers','Home Price']

#Set the different values of alpha to be tested
alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]

#Initialize the dataframe for storing coefficients.
col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,10)]
ind = ['alpha_%.2g'%alpha_ridge[i] for i in range(0,10)]
coef_matrix_ridge = pd.DataFrame(index=ind, columns=col)

models_to_plot = {1e-15:231, 1e-10:232, 1e-4:233, 1e-3:234, 1e-2:235, 5:236}
for i in range(10):
    coef_matrix_ridge.iloc[i,] = r.ridge_regression(loan_df, predictors, alpha_ridge[i], models_to_plot)
    
#Set the display format to be scientific for ease of analysis
# pd.options.display.float_format = '{:,.2g}'.format
# coef_matrix_ridge



'''
### Lasso regression
#Initialize predictors to all 15 powers of x
predictors=['credit_score','first_time_homebuyer_flag','original_combined_ltv','original_upb','original_ltv','original_interest_rate','prepayment_penalty_flag','number_of_borrowers','Home Price']


#Define the alpha values to test
alpha_lasso = [1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 1, 5, 10]

#Initialize the dataframe to store coefficients
col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,10)]
ind = ['alpha_%.2g'%alpha_lasso[i] for i in range(0,10)]
coef_matrix_lasso = pd.DataFrame(index=ind, columns=col)

#Define the models to plot
models_to_plot = {1e-10:231, 1e-5:232,1e-4:233, 1e-3:234, 1e-2:235, 1:236}

#Iterate over the 10 alpha values:
for i in range(10):
    coef_matrix_lasso.iloc[i,] = lasso_regression(loan_df, predictors, alpha_lasso[i], models_to_plot)
'''

#Import Library
#Import other necessary libraries like pandas, numpy...

#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create tree object 
model = tree.DecisionTreeClassifier(criterion='gini') # for classification, here you can change the algorithm as gini or entropy (information gain) by default it is gini  
# model = tree.DecisionTreeRegressor() for regression
# Train the model using the training sets and check score
model.fit(X, y)
model.score(X, y)
#Predict Output
predicted= model.predict(x_test)