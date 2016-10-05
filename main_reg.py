# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 15:37:58 2016

@author: Chandler
"""
import os
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
import scipy.optimize as spo

def Add_Prepay_Col_Identifiers(loan_df_train, loan_df_test):
    ### Add prepayment column to show wether there is prepayment or not
    loan_df_train['Prepayment'] = 1*(loan_df_train['Prepay Percent']>=0.2)
    loan_df_test['Prepayment'] = 1*(loan_df_test['Prepay Percent']>=0.2)

    ### Add big_prepayment column to show wether there is big prepayment or not
    loan_df_train['big_Prepayment'] = 1*(loan_df_train['Prepay Percent']>0.8)
    loan_df_test['big_Prepayment'] = 1*(loan_df_test['Prepay Percent']>0.8)

    return loan_df_train, loan_df_test

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
    """print('[Decision Tree Result] Accuracy for judging %s or not is %s' %(y,accuracy))
    print('[Decision Tree Result]  True Negative (Predict no prepayment and no prepayment) for given no %s judgement is %s' %(y,TN))
    print('[Decision Tree Result]  False Negative (Predict prepayment but no prepayment) for given no %s judgement is %s' %(y,FN))
    print('[Decision Tree Result]  True Positive (Predict prepayment and has prepayment) for given no %s judgement is %s' %(y,TP))
    print('[Decision Tree Result]  False Positive (Predict no prepayment but has prepayment) for given no %s judgement is %s' %(y,FP))
    print('\n')"""
    return df_test
        
def Logistic_Optimizer(alpha, penalty, df_train, df_test, y, y_pred, predictors):
    """
        Calculate true (false) positive (negative) prediction probabilities over the 
        dataset for a given alpha.
        Return the sum of the true positive and true negative probabilities.
    """
    logreg = LogisticRegression(penalty = penalty, C=alpha[0])
    logreg.fit(df_train[predictors],df_train[y])
    predicted = logreg.predict(df_test[predictors])
    
    # Add new prediction into testing dataset
    df_test[y_pred] = predicted

    # Calculate the accuracy
    try:    
        TN = sum(df_test[y_pred][df_test[y_pred] ==0] == df_test[y][df_test[y_pred] ==0] )/\
            len(df_test[y_pred][df_test[y_pred] ==0])
    except:
        TN = 0
    try: 
        FN = sum(df_test[y_pred][df_test[y_pred] ==0] != df_test[y][df_test[y_pred] ==0] )/\
            len(df_test[y_pred][df_test[y_pred] ==0])
    except:
        FN = 0
    try: 
        TP = sum(df_test[y_pred][df_test[y_pred] ==1] == df_test[y][df_test[y_pred] ==1] )/\
            len(df_test[y_pred][df_test[y_pred] ==1])
    except:
        TP = 0
    try: 
        FP = sum(df_test[y_pred][df_test[y_pred] ==1] != df_test[y][df_test[y_pred] ==1] )/\
            len(df_test[y_pred][df_test[y_pred] ==1])
    except:
        FP = 0

    return (-1) * (TN + TP)

def Logistic_Prediction(alpha, penalty, df_train, df_test, y, y_pred, predictors):
    logreg = LogisticRegression(penalty = penalty, C=alpha[0])
    logreg.fit(df_train[predictors],df_train[y])
    predicted = logreg.predict(df_test[predictors])
    
    # Add new prediction into testing dataset
    df_test[y_pred] = predicted

    # Calculate the accuracy
    try:    
        TN = sum(df_test[y_pred][df_test[y_pred] ==0] == df_test[y][df_test[y_pred] ==0] )/\
            len(df_test[y_pred][df_test[y_pred] ==0])
    except:
        #TN  = 'NaN'
        TN = -1
    try: 
        FN = sum(df_test[y_pred][df_test[y_pred] ==0] != df_test[y][df_test[y_pred] ==0] )/\
            len(df_test[y_pred][df_test[y_pred] ==0])
    except:
        #FN  = 'NaN' 
        FN = -1
    try: 
        TP = sum(df_test[y_pred][df_test[y_pred] ==1] == df_test[y][df_test[y_pred] ==1] )/\
            len(df_test[y_pred][df_test[y_pred] ==1])
    except:
        #TP  = 'NaN'
        TP = -1
    try: 
        FP = sum(df_test[y_pred][df_test[y_pred] ==1] != df_test[y][df_test[y_pred] ==1] )/\
            len(df_test[y_pred][df_test[y_pred] ==1])
    except:
        #FP  = 'NaN'
        FP = -1
    print('Alpha is %s' %alpha)
    print('[Logistic Regression Result]  True Negative (Predict no prepayment \
            and no prepayment) for given no %s judgement is %s' %(y,TN))
    print('[Logistic Regression Result]  False Negative (Predict prepayment \
            but no prepayment) for given no %s judgement is %s' %(y,FN))
    print('[Logistic Regression Result]  True Positive (Predict prepayment \
            and has prepayment) for given no %s judgement is %s' %(y,TP))
    print('[Logistic Regressione Result]  False Positive (Predict no prepayment \
            but has prepayment) for given no %s judgement is %s' %(y,FP))    
    print('\n')
    return df_test

def Optimize_Logistic(loan_df_train, loan_df_test, new_predictors_log):
    """ 
        Optimize logistic regression by minimizing the negative of the sum of the 
        true negative and true positive probabilities. Minimize by changing alpha.
        Return optimal alpha.
    """

    print('####### Logistic Regression #######')
    # initially ran minimization with larger values, then narrowed to range for quicker results
    alpha_list = np.arange(0.01, 0.1, 0.005) 

    alpha_best = 0
    bnds = ((0, None),)
    max_result = np.Inf
    for alpha in alpha_list:
        result = spo.minimize(Logistic_Optimizer, alpha, 
                     args=('l1',loan_df_train, loan_df_test, 'Prepayment', 
                           'Pred_Prepayment_log', new_predictors_log),
                     bounds=bnds,
                     method='SLSQP', 
                     tol=10e-5)

        if result.fun < max_result:
            max_result = result.fun
            alpha_best = result.x

    print ('Best result: ', max_result, '\nalpha:', alpha_best)
    return alpha_best

def RL_Test_Set(loan_df_train, loan_df_test, predictors, alpha_train, func, regress_func):
    """ 
        regress_func = Lasso and Ridge.
        func = r.RL_regression.

        Get subset of cols for ridge regressions for initial alphas.
        Set tolerance of not including column to less than 1e-4.
    """
    cols = []
    df_regress = pd.DataFrame(index=alpha_train, columns=predictors)

    # get results of logistic regression over range of alphas
    for alpha in alpha_train:
        # func = r.RL_regression
        result, rss = func(loan_df_train, 'Prepay Percent', predictors, alpha, regress_func)

        df_regress.ix[alpha] = result

    df_regress[df_regress < 1e-4] = 0   # values < 1e^-4 = zero : threshold for ridge

    return df_regress

def RL_Optimizer(loan_df_train, loan_df_test, predictors, alpha_train, func, optimize_func, 
                 regress_func):
    """ 
        Lasso and Ridge.
        First get number of columns to consider over a range of alphas using train set.
        Using the columns selected for each alpha, minimize rss over test set.
        Minimize the squared errors of ridge regression by finding the optimal alpha.
        Return the optimal alpha.
    """
    df_regress = RL_Test_Set(loan_df_train, loan_df_test, predictors, alpha_train, func, regress_func)
    
    # optimization initial parameters
    optimal_alpha = 0
    ridge_cols = []     # optimal subset of cols to return
    min_rss = np.Inf
    bnds = ((0, None),)

    for alpha, row in df_regress.iterrows():
        # Get index of all nonzeo columns from row of logistic reg dataframe
        # Select subset of loan_df_test from index
        for i in range(len(row)):
            idx = [idx for idx in range(len(row)) if row[idx] > 0]  # select nonzero indexes
            test_predictors = loan_df_test[predictors].columns[idx] # get nonzero idx columns

        result = spo.minimize(optimize_func, alpha, 
                         args=(loan_df_test, 'Prepay Percent', 
                               test_predictors, regress_func),
                         bounds=bnds,
                         method='SLSQP', 
                         tol=10e-5)

        # Select min rss over test set
        if (result.fun < min_rss):
            min_rss = result.fun
            optimal_alpha = alpha
            ridge_cols = test_predictors

    #print (optimal_alpha, min_rss, ridge_cols)
    return ridge_cols

def rest(loan_df_train, loan_df_test, predictors, target_variable, new_predictors_log):
    ####### Ridge ############
    alpha_ridge_train = np.random.random(20)
    ridge_cols = RL_Optimizer(loan_df_train, loan_df_test, predictors, alpha_ridge_train,
                                r.RL_regression, r.RL_optimizer, Ridge)

    print ('Ridge columns: ')
    for col in ridge_cols:
        print (col)
    ###########################

    ####### Lasso ############
    alpha_lasso_train = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
    lasso_cols = RL_Optimizer(loan_df_train, loan_df_test, predictors, alpha_lasso_train,
                                r.RL_regression, r.RL_optimizer, Lasso)

    print ('\nLasso columns: ')
    for col in lasso_cols:
        print (col)    
    ##########################

    # Calculate the Accuracy
    print('####### Two Step Logistic + Decision Tree #######')
    decision_tree(loan_df_train, loan_df_test, new_predictors_log, 'gini', 'big_Prepayment', 
                 'Pred_big_prepayment_log') 

    # Generate testing dataset which doesn't have big prepayment according to algo_1_ridge
    loan_df_test_small_log = loan_df_test[loan_df_test['Pred_big_prepayment_log'] == 0]


    ############ Decision Tree_2 for finding low prepayment rate loan among the group with <80% 
    ############ prepayment percent using ridge regression result ############
    ### Filter out the prepay percent only with <80% and develop algo_2
    loan_df_train_small = loan_df_train[loan_df_train['Prepay Percent']<0.8]

    # Calculate the Accuracy
    decision_tree(loan_df_train_small, loan_df_test_small_log, new_predictors_log, 'gini', 
                 'Prepayment', 'Pred_prepayment_ridge') 
     
     
    ############ Decision Tree_1 for divide higher than 80% and less then 80% using ridge 
    ############ regression result ############
    new_predictors_ridge = ['mortgage_insurance_percentage','original_combined_ltv','original_ltv',
                            'original_interest_rate','prepayment_penalty_flag','number_of_borrowers']
    # Calculate the Accuracy
    print('####### Two Step Ridge + Decision Tree #######')
    decision_tree(loan_df_train, loan_df_test, new_predictors_ridge, 'gini', 'big_Prepayment', 
                  'Pred_big_prepayment_ridge') 

    # Generate testing dataset which doesn't have big prepayment according to algo_1_ridge
    loan_df_test_small_ridge = loan_df_test[loan_df_test['Pred_big_prepayment_ridge'] == 0]


    ############ Decision Tree_2 for finding low prepayment rate loan among the group with 
    ############ <80% prepayment percent using ridge regression result ############
    ### Filter out the prepay percent only with <80% and develop algo_2
    loan_df_train_small = loan_df_train[loan_df_train['Prepay Percent']<0.8]

    # Calculate the Accuracy
    decision_tree(loan_df_train_small, loan_df_test_small_ridge, new_predictors_ridge, 'gini', 
                 'Prepayment', 'Pred_prepayment_ridge')



    ############ Decision Tree_1 for divide higher than 80% and less then 80% using lasso 
    ############ regression result ############
    new_predictors_lasso = ['mortgage_insurance_percentage','original_ltv','original_interest_rate',
                            'prepayment_penalty_flag','number_of_borrowers']
    # Calculate the Accuracy
    print('####### Two Step Lasso + Decision Tree #######')
    decision_tree(loan_df_train, loan_df_test, new_predictors_lasso, 'gini', 'big_Prepayment', 
                  'Pred_big_prepayment_lasso') 

    # Generate testing dataset which doesn't have big prepayment according to algo_1_ridge
    loan_df_test_small_lasso = loan_df_test[loan_df_test['Pred_big_prepayment_lasso'] == 0]

    ############ Decision Tree_2 for finding low prepayment rate loan among the group with 
    ############ <80% prepayment percent using lasso regression result ############
    ### Filter out the prepay percent only with <80% and develop algo_2
    loan_df_train_small = loan_df_train[loan_df_train['Prepay Percent']<0.8]

    # Calculate the Accuracy
    decision_tree(loan_df_train_small, loan_df_test_small_lasso, new_predictors_lasso, 'gini', 
                  'Prepayment', 'Pred_prepayment_ridge')

    ############ Logistic regression for predicting low prepayment or high prepayment 5% 
    ############ benchmark ############
    new_predictors_log = ['mortgage_insurance_percentage','original_loan_term','credit_score',
                        'original_ltv','prepayment_penalty_flag','number_of_borrowers']
    # Calculate the Accuracy

    ###### Logistic ##############
    alpha_logistic = Optimize_Logistic(loan_df_train, loan_df_test, new_predictors_log)

    Logistic_Prediction(alpha_logistic, 'l1',loan_df_train, loan_df_test, 'Prepayment', 
                           'Pred_Prepayment_log', new_predictors_log)

    logistic_result = r.logistic_regression('l1', loan_df_train, 'Prepayment', 
                                            predictors, alpha_logistic)
    ############################

if __name__=="__main__":
    data_file_path = r'C:\Users\Alex\Desktop\abs_data\\' # modify to own path
    ############ show histogram of different variable ############ 
    target_variable = ['credit_score','first_time_homebuyer_flag','original_combined_ltv',
                       'original_upb','original_ltv','original_interest_rate',
                       'prepayment_penalty_flag','number_of_borrowers','Prepay Percent','Home Price']
    models_to_plot = {'credit_score':251,'first_time_homebuyer_flag':252,
                      'original_combined_ltv':253,'original_upb':254,
                      'original_ltv':255,'original_interest_rate':256,
                      'prepayment_penalty_flag':257,'number_of_borrowers':258,
                      'Prepay Percent':259,'Home Price':2510}

    ############  Logistic & Ridge regression ############ 
    predictors=['mortgage_insurance_percentage','original_loan_term','credit_score',
                'first_time_homebuyer_flag','original_combined_ltv','original_upb',
                'original_ltv','original_interest_rate','prepayment_penalty_flag',
                'number_of_borrowers','Home Price']

    loan_df = pd.read_csv(data_file_path + 'freddie_final.csv')
    loan_df = loan_df[['loan_seq_number','Prepay Percent','mortgage_insurance_percentage',
                       'original_loan_term','credit_score','first_time_homebuyer_flag',
                       'original_combined_ltv','original_upb','original_ltv',
                       'original_interest_rate','prepayment_penalty_flag',
                       'number_of_borrowers','Home Price']] 
    ############ Decision Tree_1 for divide higher than 80% and less then 80% 
    ###          using logistic regression result ############
    new_predictors_log = ['mortgage_insurance_percentage','original_loan_term',
                          'credit_score','original_ltv','prepayment_penalty_flag',
                          'number_of_borrowers']

    # Drop all NAs
    loan_df = pd.DataFrame.dropna(loan_df)

    # Pick up the training set and testing set - total # of rows currently = 9742
    # train = 7000; test = 2742
    random.seed(0)
    sample_index = random.sample(list(loan_df.index), 7000)
    loan_df_train = loan_df.ix[sample_index] 
    loan_df_test = loan_df.drop(sample_index)

    loan_df_train, loan_df_test = Add_Prepay_Col_Identifiers(loan_df_train, loan_df_test)

    rest(loan_df_train, loan_df_test, predictors, target_variable, new_predictors_log)













