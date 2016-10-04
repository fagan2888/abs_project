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
from sklearn.linear_model import LogisticRegression
from sklearn import tree
import regression as r
import random



loan_df = pd.read_csv('C:/Users/Chandler/Desktop/Berkeley/2016 Fall/230M ABS/abs_project/freddie_final_2.csv')
loan_df = loan_df[['loan_seq_number','Prepay Percent','mortgage_insurance_percentage','original_loan_term','credit_score','first_time_homebuyer_flag','original_combined_ltv','original_upb','original_ltv','original_interest_rate','prepayment_penalty_flag','number_of_borrowers','Home Price']]

# Drop all NAs
loan_df = pd.DataFrame.dropna(loan_df)

# Pick up the training set and testing set
random.seed(0)
sample_index = random.sample(list(loan_df.index), 7000)
loan_df_train = loan_df.ix[sample_index]
loan_df_test = loan_df.drop(sample_index)



############ show histogram of different variable ############ 
target_variable = ['credit_score','first_time_homebuyer_flag','original_combined_ltv','original_upb','original_ltv','original_interest_rate','prepayment_penalty_flag','number_of_borrowers','Prepay Percent','Home Price']
models_to_plot = {'credit_score':251,'first_time_homebuyer_flag':252,'original_combined_ltv':253,'original_upb':254,'original_ltv':255,'original_interest_rate':256,'prepayment_penalty_flag':257,'number_of_borrowers':258,'Prepay Percent':259,'Home Price':2510}

# r.variable_dist(loan_df, target_variable, models_to_plot)



### Add prepayment column to show wether there is prepayment or not
loan_df_train['Prepayment'] = 1*(loan_df_train['Prepay Percent']>=0.2)
loan_df_test['Prepayment'] = 1*(loan_df_test['Prepay Percent']>=0.2)

### Add big_prepayment column to show wether there is big prepayment or not
loan_df_train['big_Prepayment'] = 1*(loan_df_train['Prepay Percent']>0.8)
loan_df_test['big_Prepayment'] = 1*(loan_df_test['Prepay Percent']>0.8)



############ Logistic regression ############
predictors=['mortgage_insurance_percentage','original_loan_term','credit_score','first_time_homebuyer_flag','original_combined_ltv','original_upb','original_ltv','original_interest_rate','prepayment_penalty_flag','number_of_borrowers','Home Price']

#Set the different values of alpha to be tested
C = [1e-4, 1e-3, 5e-2, 1e-2, 0.05, 0.1, 0.5, 1, 2, 5]

#Initialize the dataframe for storing coefficients.
col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,12)]
ind = ['alpha_%.2g'%C[i] for i in range(0,10)]
coef_matrix_log = pd.DataFrame(index=ind, columns=col)

models_to_plot = {1e-15:231, 1e-10:232, 1e-4:233, 1e-3:234, 1e-2:235, 5:236}
for i in range(10):
    coef_matrix_log.iloc[i,] = r.logistic_regression('l1', loan_df_train, 'Prepayment', predictors, C[i], models_to_plot)
    

############  Ridge regression ############ 
predictors=['mortgage_insurance_percentage','original_loan_term','credit_score','first_time_homebuyer_flag','original_combined_ltv','original_upb','original_ltv','original_interest_rate','prepayment_penalty_flag','number_of_borrowers','Home Price']

#Set the different values of alpha to be tested
alpha_ridge = [1e-4, 1e-3, 5e-2, 1e-2, 0.05, 0.1, 0.5, 1, 2, 5]

#Initialize the dataframe for storing coefficients.
col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,12)]
ind = ['alpha_%.2g'%alpha_ridge[i] for i in range(0,10)]
coef_matrix_ridge = pd.DataFrame(index=ind, columns=col)

models_to_plot = {1e-4:231, 1e-2:232, 0.05:233, 0.1:234, 0.5:235, 1:236}
for i in range(10):
    coef_matrix_ridge.iloc[i,] = r.ridge_regression(loan_df_train, 'Prepay Percent', predictors, alpha_ridge[i], models_to_plot)
    
#Set the display format to be scientific for ease of analysis
# pd.options.display.float_format = '{:,.2g}'.format
# coef_matrix_ridge



############ Lasso regression ############
#Initialize predictors to all 15 powers of x
predictors=['mortgage_insurance_percentage','original_loan_term','credit_score','first_time_homebuyer_flag','original_combined_ltv','original_upb','original_ltv','original_interest_rate','prepayment_penalty_flag','number_of_borrowers','Home Price']


#Define the alpha values to test
alpha_lasso = [1e-8, 1e-7, 1e-6, 1e-5 ,1e-4, 1e-3, 5e-2, 1e-2, 0.1, 0.5]

#Initialize the dataframe to store coefficients
col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,12)]
ind = ['alpha_%.2g'%alpha_lasso[i] for i in range(0,10)]
coef_matrix_lasso = pd.DataFrame(index=ind, columns=col)

#Define the models to plot
models_to_plot = {1e-8:231, 1e-5:232, 1e-3:233, 1e-2:234, 0.1:235, 0.5:236}

#Iterate over the 10 alpha values:
for i in range(10):
    coef_matrix_lasso.iloc[i,] = r.lasso_regression(loan_df_train, 'Prepay Percent', predictors, alpha_lasso[i], models_to_plot)



def decision_tree(df_train, df_test, predictors, method, y, y_pred):
    #Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
    # Create tree object 
    model = tree.DecisionTreeClassifier(criterion = method) # for classification, here you can change the algorithm as gini or entropy (information gain) by default it is gini  
    # model = tree.DecisionTreeRegressor() for regression
    # Train the model using the training sets and check score
    model.fit(df_train[predictors], df_train[y])
    model.score(df_train[predictors], df_train[y])
    #Predict Output
    predicted = model.predict(df_test[predictors])
    df_test[y_pred] = predicted
    
    # Calculate the accuracy
    accuracy = sum(df_test[y_pred] == df_test[y])/len(df_test[y_pred])
    TN = sum(df_test[y_pred][df_test[y_pred] ==0] == df_test[y][df_test[y_pred] ==0] )/len(df_test[y_pred][df_test[y_pred] ==0])
    FN = sum(df_test[y_pred][df_test[y_pred] ==0] != df_test[y][df_test[y_pred] ==0] )/len(df_test[y_pred][df_test[y_pred] ==0])
    
    TP = sum(df_test[y_pred][df_test[y_pred] ==1] == df_test[y][df_test[y_pred] ==1] )/len(df_test[y_pred][df_test[y_pred] ==1])
    FP = sum(df_test[y_pred][df_test[y_pred] ==1] != df_test[y][df_test[y_pred] ==1] )/len(df_test[y_pred][df_test[y_pred] ==1])
    print('[Decision Tree Result] Accuracy for judging %s or not is %s' %(y,accuracy))
    print('[Decision Tree Result]  True Negative (Predict no prepayment and no prepayment) for given no %s judgement is %s' %(y,TN))
    print('[Decision Tree Result]  False Negative (Predict prepayment but no prepayment) for given no %s judgement is %s' %(y,FN))
    print('[Decision Tree Result]  True Positive (Predict prepayment and has prepayment) for given no %s judgement is %s' %(y,TP))
    print('[Decision Tree Result]  False Positive (Predict no prepayment but has prepayment) for given no %s judgement is %s' %(y,FP))
    print('\n')
    return df_test
    
    
    

############ Decision Tree_1 for divide higher than 80% and less then 80% using logistic regression result ############
new_predictors_log = ['mortgage_insurance_percentage','original_loan_term','credit_score','original_ltv','prepayment_penalty_flag','number_of_borrowers']
# Calculate the Accuracy
print('####### Two Step Logistic + Decision Tree #######')
decision_tree(loan_df_train, loan_df_test, new_predictors_log, 'gini', 'big_Prepayment', 'Pred_big_prepayment_log') 

# Generate testing dataset which doesn't have big prepayment according to algo_1_ridge
loan_df_test_small_log = loan_df_test[loan_df_test['Pred_big_prepayment_log'] == 0]


############ Decision Tree_2 for finding low prepayment rate loan among the group with <80% prepayment percent using ridge regression result ############
### Filter out the prepay percent only with <80% and develop algo_2
loan_df_train_small = loan_df_train[loan_df_train['Prepay Percent']<0.8]

# Calculate the Accuracy
decision_tree(loan_df_train_small, loan_df_test_small_log, new_predictors_log, 'gini', 'Prepayment', 'Pred_prepayment_ridge') 
 
 
 
 
############ Decision Tree_1 for divide higher than 80% and less then 80% using ridge regression result ############
new_predictors_ridge = ['mortgage_insurance_percentage','original_combined_ltv','original_ltv','original_interest_rate','prepayment_penalty_flag','number_of_borrowers']
# Calculate the Accuracy
print('####### Two Step Ridge + Decision Tree #######')
decision_tree(loan_df_train, loan_df_test, new_predictors_ridge, 'gini', 'big_Prepayment', 'Pred_big_prepayment_ridge') 

# Generate testing dataset which doesn't have big prepayment according to algo_1_ridge
loan_df_test_small_ridge = loan_df_test[loan_df_test['Pred_big_prepayment_ridge'] == 0]


############ Decision Tree_2 for finding low prepayment rate loan among the group with <80% prepayment percent using ridge regression result ############
### Filter out the prepay percent only with <80% and develop algo_2
loan_df_train_small = loan_df_train[loan_df_train['Prepay Percent']<0.8]

# Calculate the Accuracy
decision_tree(loan_df_train_small, loan_df_test_small_ridge, new_predictors_ridge, 'gini', 'Prepayment', 'Pred_prepayment_ridge')



############ Decision Tree_1 for divide higher than 80% and less then 80% using lasso regression result ############
new_predictors_lasso = ['mortgage_insurance_percentage','original_ltv','original_interest_rate','prepayment_penalty_flag','number_of_borrowers']
# Calculate the Accuracy
print('####### Two Step Lasso + Decision Tree #######')
decision_tree(loan_df_train, loan_df_test, new_predictors_lasso, 'gini', 'big_Prepayment', 'Pred_big_prepayment_lasso') 

# Generate testing dataset which doesn't have big prepayment according to algo_1_ridge
loan_df_test_small_lasso = loan_df_test[loan_df_test['Pred_big_prepayment_lasso'] == 0]

############ Decision Tree_2 for finding low prepayment rate loan among the group with <80% prepayment percent using lasso regression result ############
### Filter out the prepay percent only with <80% and develop algo_2
loan_df_train_small = loan_df_train[loan_df_train['Prepay Percent']<0.8]

# Calculate the Accuracy
decision_tree(loan_df_train_small, loan_df_test_small_lasso, new_predictors_lasso, 'gini', 'Prepayment', 'Pred_prepayment_ridge')







'''
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








'''

### IF no two step algo, just use whole training set
###### Decision Tree_2 for finding low prepayment rate loan among the group with <80% prepayment percent using ridge regression result ######
### Filter out the prepay percent only with <80% and develop algo_2


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
condi_accuracy_2_ridge = sum(loan_df_test['Pred_prepayment_ridge'][loan_df_test['Pred_prepayment_ridge'] ==0] == loan_df_test['Prepayment'][loan_df_test['Pred_prepayment_ridge'] ==0] )/len(loan_df_test['Pred_prepayment_ridge'][loan_df_test['Pred_prepayment_ridge'] ==0])
print('[Ridge regression] Accuracy for judging prepayment or not is %s' %accuracy_2_ridge)
print('[Ridge regression] Conditional accuracy for given prepayment judgement is %s' %condi_accuracy_2_ridge)


'''

def Logistic_prediction(penalty, df_train, df_test, y, y_pred, predictors, alpha):
    # fit regression
    logreg = LogisticRegression(penalty = penalty, C=alpha)
    logreg.fit(df_train[predictors],df_train[y])
    predicted = logreg.predict(df_test[predictors])
    
    # Add new prediction into testing dataset
    df_test[y_pred] = predicted

    # Calculate the accuracy
    try:    
        TN = sum(df_test[y_pred][df_test[y_pred] ==0] == df_test[y][df_test[y_pred] ==0] )/len(df_test[y_pred][df_test[y_pred] ==0])
    except:
        TN  = 'NaN'
    try: 
        FN = sum(df_test[y_pred][df_test[y_pred] ==0] != df_test[y][df_test[y_pred] ==0] )/len(df_test[y_pred][df_test[y_pred] ==0])
    except:
        FN  = 'NaN' 
    try: 
        TP = sum(df_test[y_pred][df_test[y_pred] ==1] == df_test[y][df_test[y_pred] ==1] )/len(df_test[y_pred][df_test[y_pred] ==1])
    except:
        TP  = 'NaN'
    try: 
        FP = sum(df_test[y_pred][df_test[y_pred] ==1] != df_test[y][df_test[y_pred] ==1] )/len(df_test[y_pred][df_test[y_pred] ==1])
    except:
        FP  = 'NaN'
    print('Alpha is %s' %alpha)
    print('[Logistic Regression Result]  True Negative (Predict no prepayment and no prepayment) for given no %s judgement is %s' %(y,TN))
    print('[Logistic Regression Result]  False Negative (Predict prepayment but no prepayment) for given no %s judgement is %s' %(y,FN))
    print('[Logistic Regression Result]  True Positive (Predict prepayment and has prepayment) for given no %s judgement is %s' %(y,TP))
    print('[Logistic Regressione Result]  False Positive (Predict no prepayment but has prepayment) for given no %s judgement is %s' %(y,FP))    
    print('\n')    
    return df_test
    
    

############ Logistic regression for predicting low prepayment or high prepayment 5% benchmark ############
new_predictors_log = ['mortgage_insurance_percentage','original_loan_term','credit_score','original_ltv','prepayment_penalty_flag','number_of_borrowers']
# Calculate the Accuracy
print('####### Logistic Regression #######')
alpha_list = [1e-8, 1e-7, 1e-6, 1e-5 ,1e-4, 1e-3, 5e-2, 1e-2, 0.1, 0.5, 1, 2, 5]

for alpha in alpha_list:
    Logistic_prediction('l1',loan_df_train, loan_df_test, 'Prepayment', 'Pred_Prepayment_log', new_predictors_log, alpha) 
    # print(alpha)



