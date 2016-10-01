# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 15:29:29 2016

@author: Chandler
"""
import pandas as pd 
import numpy as np 
import dateutil.relativedelta
pd.options.mode.chained_assignment = None  # default='warn'

def Merge_Orig_Perf(df_orig, df_perf):
	df_orig_temp = df_orig[['loan_seq_number', 'original_interest_rate', 'original_upb']]
	df_orig_temp.rename(columns={'original_interest_rate':'current_interest_rate'}, inplace=True)
	df_orig_temp.rename(columns={'original_upb':'current_actual_upb'}, inplace=True)

	df_merged = pd.concat([df_perf, df_orig_temp])
	df_merged['reporting_period'] = pd.to_datetime(df_merged['reporting_period'], format='%m/%d/%Y',
												  errors='ignore')
	df_merged['loan_age'].fillna(-1, inplace=True)
	df_merged.sort_values(by=['loan_seq_number', 'loan_age'], ascending=[True,True], 
						inplace=True)
	df_merged.fillna(np.nan, inplace=True)
	#df_merged.to_csv('freddie_merged.csv', index=False)

	return df_merged

def Remove_Missing_Principal_Bonds(df_merged, df_orig):
	"""
		Bonds who were missing current principal balance midway through.
	"""
	df_merged.set_index('loan_seq_number', inplace=True)
	df_orig.set_index('loan_seq_number', inplace=True)
	index_to_drop = df_merged[df_merged['current_actual_upb'] == 'nan'].index
	df_merged.drop(index_to_drop, inplace=True)
	df_orig.drop(index_to_drop, inplace=True)
	df_merged.reset_index(level=0, inplace=True)	# reset 'loan_seq_number' to column
	df_orig.reset_index(level=0, inplace=True)	# reset 'loan_seq_number' to column

	return df_merged, df_orig

def Create_Pay_Cols(df_mtx):
	"""
		Columns: [0:'loan_seq_number', 1:'current_actual_upb', 2:'current_interest_rate',
				 3:'delinquency_status', 4:'loan_age', 5:'remaining_months_to_maturity',
				 6:'reporting_period']
	"""
	# columns added
	cols_added = ['Principal+Interest Paid', 'Interest Paid','Principal Paid', 
				  'Principal+Prepayment Paid', 'Prepayment Paid']

	# convert interest rate to monthly decimals
	df_mtx[:, 2] = df_mtx[:, 2] / 100.0 / 12.0

	int_pay_arr = [0] * df_mtx.shape[0]
	amort_pay_arr = [0] * df_mtx.shape[0]
	princ_arr = [0] * df_mtx.shape[0]
	princ_and_ppy_arr = [0] * df_mtx.shape[0]
	prepay_arr = [0] * df_mtx.shape[0]

	for i in range(df_mtx.shape[0]):
		if df_mtx[i, 5] == 'nan':	# remaining_months_to_maturity
			df_mtx[i, 5] = int(df_mtx[i+1, 5]) + 1	# fill in original r_m_t_m
		else:
			int_pay_arr[i] = df_mtx[i-1, 1] * df_mtx[i, 2]	# 'interest_paid' column
			amort_pay_arr[i] = (-1)*np.pmt(df_mtx[i, 2], df_mtx[i, 5], df_mtx[i-1, 1])
			princ_arr[i] = amort_pay_arr[i] - int_pay_arr[i]
			princ_and_ppy_arr[i] = df_mtx[i-1, 1] - df_mtx[i, 1]
			prepay_arr[i] = princ_and_ppy_arr[i] - princ_arr[i]

	df_new_mtx = np.column_stack([df_mtx, amort_pay_arr, int_pay_arr, princ_arr, 
								  princ_and_ppy_arr, prepay_arr])

	return df_new_mtx, cols_added

def Calc_Prepay_Percent(df, df_orig):
	"""
		Group performance data by 'loan_seq_number'.
		Take sum of all prepayments, including negative prepayemnts.
		Divide original loan amount by sum of prepayments to get prepayment %.
	"""
	grouped = df.groupby('loan_seq_number')
	prepay_percent_arr = []

	for name, group in grouped:
		prepay_percent_arr.append(
			[name, group['Prepayment Paid'].sum() / group['current_actual_upb'].max()])

	prepay_percent_arr = np.array(prepay_percent_arr)
	df_orig['Prepay Percent'] = prepay_percent_arr[:,1]

	return df_orig
def Revise_format(df_orig):
    '''
    Change first_time_homebuyer_flag and prepayment_penalty_flag into dummy variable, change ltv and int rate into decimal number
    '''
    for i in range(len(df_orig['first_time_homebuyer_flag'])):
        df_orig['first_time_homebuyer_flag'][i] = 1 if df_orig['first_time_homebuyer_flag'][i] == 'Y' else 0
        df_orig['prepayment_penalty_flag'][i] = 1 if df_orig['prepayment_penalty_flag'][i] == 'Y' else 0        
    df_orig['original_combined_ltv'] = df_orig['original_combined_ltv']/100
    df_orig['original_interest_rate'] = df_orig['original_interest_rate']/100
    df_orig['original_ltv'] = df_orig['original_ltv']/100
    
    return df_orig