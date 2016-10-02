# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 15:37:58 2016

@author: Chandler
"""
import os
os.chdir('C:/Users/Chandler/Desktop/Berkeley/2016 Fall/230M ABS/abs_project')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
import regression as r

loan_df_train = pd.read_csv('C:/Users/Chandler/Desktop/Berkeley/2016 Fall/230M ABS/abs_project/freddie_final.csv')
loan_df_test = pd.read_csv('C:/Users/Chandler/Desktop/Berkeley/2016 Fall/230M ABS/abs_project/freddie_final_2.csv')

# delete the first 1000 duplicated rows
loan_df_test = loan_df_test.iloc[982:]

# delete the blank cell
loan_df_test = loan_df_test[~np.isnan(loan_df_test['number_of_borrowers'])]

###### show histogram of different variable ###### 
target_variable = ['credit_score','first_time_homebuyer_flag','original_combined_ltv','original_upb','original_ltv','original_interest_rate','prepayment_penalty_flag','number_of_borrowers','Prepay Percent','Home Price']
models_to_plot = {'credit_score':251,'first_time_homebuyer_flag':252,'original_combined_ltv':253,'original_upb':254,'original_ltv':255,'original_interest_rate':256,'prepayment_penalty_flag':257,'number_of_borrowers':258,'Prepay Percent':259,'Home Price':2510}

# r.variable_dist(loan_df, target_variable, models_to_plot)



### Add prepayment column to show wether there is prepayment or not
loan_df_train['Prepayment'] = 1*(loan_df_train['Prepay Percent']<=0.05)
loan_df_test['Prepayment'] = 1*(loan_df_test['Prepay Percent']<=0.05)

### Add big_prepayment column to show wether there is big prepayment or not
loan_df_train['big_Prepayment'] = 1*(loan_df_train['Prepay Percent']>0.8)
loan_df_test['big_Prepayment'] = 1*(loan_df_test['Prepay Percent']>0.8)


'''
######  ridge regression ###### 
predictors=['credit_score','first_time_homebuyer_flag','original_combined_ltv','original_upb','original_ltv','original_interest_rate','prepayment_penalty_flag','number_of_borrowers','Home Price']

#Set the different values of alpha to be tested
alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]

#Initialize the dataframe for storing coefficients.
col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,10)]
ind = ['alpha_%.2g'%alpha_ridge[i] for i in range(0,10)]
coef_matrix_ridge = pd.DataFrame(index=ind, columns=col)

models_to_plot = {1e-15:231, 1e-10:232, 1e-4:233, 1e-3:234, 1e-2:235, 5:236}
for i in range(10):
    coef_matrix_ridge.iloc[i,] = r.ridge_regression(loan_df_train, 'Prepay Percent', predictors, alpha_ridge[i], models_to_plot)
    
#Set the display format to be scientific for ease of analysis
# pd.options.display.float_format = '{:,.2g}'.format
# coef_matrix_ridge
'''


'''
######Lasso regression ######
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
    coef_matrix_lasso.iloc[i,] = r.lasso_regression(loan_df_train, 'Prepay Percent', predictors, alpha_lasso[i], models_to_plot)
'''

###### Decision Tree_1 for divide higher than 80% and less then 80% using ridge regression result ######
new_predictors_1_ridge = ['first_time_homebuyer_flag','original_combined_ltv','original_ltv','original_interest_rate','prepayment_penalty_flag']

#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create tree object 
model_1_ridge = tree.DecisionTreeClassifier(criterion='gini') # for classification, here you can change the algorithm as gini or entropy (information gain) by default it is gini  
# model = tree.DecisionTreeRegressor() for regression
# Train the model using the training sets and check score
model_1_ridge.fit(loan_df_train[new_predictors_1_ridge], loan_df_train['big_Prepayment'])
model_1_ridge.score(loan_df_train[new_predictors_1_ridge], loan_df_train['big_Prepayment'])
#Predict Output
predicted_1_ridge = model_1_ridge.predict(loan_df_test[new_predictors_1_ridge])
loan_df_test['Pred_big_prepayment_ridge'] = predicted_1_ridge

# Calculate the accuracy
accuracy_1_ridge = sum(loan_df_test['Pred_big_prepayment_ridge'] == loan_df_test['big_Prepayment'])/len(loan_df_test['Pred_big_prepayment_ridge'])
condi_accuracy_1_ridge = sum(loan_df_test['Pred_big_prepayment_ridge'][loan_df_test['Pred_big_prepayment_ridge'] ==0] == loan_df_test['big_Prepayment'][loan_df_test['Pred_big_prepayment_ridge'] ==0] )/len(loan_df_test['Pred_big_prepayment_ridge'][loan_df_test['Pred_big_prepayment_ridge'] ==0])

print('[Ridge regression] Accuracy for judging big prepayment or not is %s' %accuracy_1_ridge)
print('[Ridge regression] Conditional accuracy for given no big prepayment judgement is %s' %condi_accuracy_1_ridge)

# Generate testing dataset which doesn't have big prepayment according to algo_1_ridge
loan_df_test_small_ridge = loan_df_test[loan_df_test['Pred_big_prepayment_ridge'] == 0]


###### Decision Tree_2 for finding low prepayment rate loan among the group with <80% prepayment percent using ridge regression result ######
### Filter out the prepay percent only with <80% and develop algo_2
loan_df_train_small = loan_df_train[loan_df_train['Prepay Percent']<0.8]

# Set predictors from ridge result
new_predictors_ridge = ['first_time_homebuyer_flag','original_combined_ltv','original_ltv','original_interest_rate','prepayment_penalty_flag']
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create tree object 
model_2_ridge = tree.DecisionTreeClassifier(criterion='gini') # for classification, here you can change the algorithm as gini or entropy (information gain) by default it is gini  
# model = tree.DecisionTreeRegressor() for regression
# Train the model using the training sets and check score
model_2_ridge.fit(loan_df_train_small[new_predictors_ridge], loan_df_train_small['Prepayment'])
model_2_ridge.score(loan_df_train_small[new_predictors_ridge], loan_df_train_small['Prepayment'])
#Predict Output
predicted_2_ridge = model_2_ridge.predict(loan_df_test_small_ridge[new_predictors_ridge])


# Calculate the accuracy
loan_df_test_small_ridge['Pred_prepayment_ridge'] = predicted_2_ridge
accuracy_2_ridge = sum(loan_df_test_small_ridge['Pred_prepayment_ridge'] == loan_df_test_small_ridge['Prepayment'])/len(loan_df_test_small_ridge['Pred_prepayment_ridge'])
condi_accuracy_2_ridge = sum(loan_df_test_small_ridge['Pred_prepayment_ridge'][loan_df_test_small_ridge['Pred_prepayment_ridge'] ==1] == loan_df_test_small_ridge['Prepayment'][loan_df_test_small_ridge['Pred_prepayment_ridge'] ==1] )/len(loan_df_test_small_ridge['Pred_prepayment_ridge'][loan_df_test_small_ridge['Pred_prepayment_ridge'] ==1])
print('[Ridge regression] Accuracy for judging prepayment or not is %s' %accuracy_2_ridge)
print('[Ridge regression] Conditional accuracy for given prepayment judgement is %s' %condi_accuracy_2_ridge)


'''
###### Decision Tree_1 for divide higher than 80% and less then 80% using lasso regression result ######
new_predictors_1_lasso = ['first_time_homebuyer_flag','original_ltv','original_interest_rate','prepayment_penalty_flag','number_of_borrowers']

#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create tree object 
model_1_lasso = tree.DecisionTreeClassifier(criterion='gini') # for classification, here you can change the algorithm as gini or entropy (information gain) by default it is gini  
# model = tree.DecisionTreeRegressor() for regression
# Train the model using the training sets and check score
model_1_lasso.fit(loan_df_train[new_predictors_1_lasso], loan_df_train['big_Prepayment'])
model_1_lasso.score(loan_df_train[new_predictors_1_lasso], loan_df_train['big_Prepayment'])
#Predict Output
predicted_1_lasso = model_1_lasso.predict(loan_df_test[new_predictors_1_lasso])

loan_df_test['Pred_big_prepayment_lasso'] = predicted_1_lasso

# Calculate the accuracy
accuracy_1_lasso = sum(loan_df_test['Pred_big_prepayment_lasso'] == loan_df_test['big_Prepayment'])/len(loan_df_test['Pred_big_prepayment_lasso'])
condi_accuracy_1_lasso = sum(loan_df_test['Pred_big_prepayment_lasso'][loan_df_test['Pred_big_prepayment_lasso'] ==0] == loan_df_test['big_Prepayment'][loan_df_test['Pred_big_prepayment_lasso'] ==0] )/len(loan_df_test['Pred_big_prepayment_lasso'][loan_df_test['Pred_big_prepayment_lasso'] ==0])

print('[Lasso regression] Accuracy for judging big prepayment or not is %s' %accuracy_1_lasso)
print('[Lasso regression] Conditional accuracy for given big prepayment judgement is %s' %condi_accuracy_1_lasso)

# Generate testing dataset which doesn't have big prepayment according to algo_1_ridge
loan_df_test_small_lasso = loan_df_test[loan_df_test['Pred_big_prepayment_ridge'] == 0]
'''










### IF no two step algo, just use whole training set
###### Decision Tree_2 for finding low prepayment rate loan among the group with <80% prepayment percent using ridge regression result ######
### Filter out the prepay percent only with <80% and develop algo_2


# Set predictors from ridge result
new_predictors_ridge = ['first_time_homebuyer_flag','original_combined_ltv','original_ltv','original_interest_rate','prepayment_penalty_flag']
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create tree object 
model_2_ridge = tree.DecisionTreeClassifier(criterion='entropy') # for classification, here you can change the algorithm as gini or entropy (information gain) by default it is gini  
# model = tree.DecisionTreeRegressor() for regression
# Train the model using the training sets and check score
model_2_ridge.fit(loan_df_train[new_predictors_ridge], loan_df_train['Prepayment'])
model_2_ridge.score(loan_df_train[new_predictors_ridge], loan_df_train['Prepayment'])
#Predict Output
predicted_2_ridge = model_2_ridge.predict(loan_df_test[new_predictors_ridge])


# Calculate the accuracy
loan_df_test['Pred_prepayment_ridge'] = predicted_2_ridge
accuracy_2_ridge = sum(loan_df_test['Pred_prepayment_ridge'] == loan_df_test['Prepayment'])/len(loan_df_test['Pred_prepayment_ridge'])
condi_accuracy_2_ridge = sum(loan_df_test['Pred_prepayment_ridge'][loan_df_test['Pred_prepayment_ridge'] ==1] == loan_df_test['Prepayment'][loan_df_test['Pred_prepayment_ridge'] ==1] )/len(loan_df_test['Pred_prepayment_ridge'][loan_df_test['Pred_prepayment_ridge'] ==1])
print('[Ridge regression] Accuracy for judging prepayment or not is %s' %accuracy_2_ridge)
print('[Ridge regression] Conditional accuracy for given prepayment judgement is %s' %condi_accuracy_2_ridge)












