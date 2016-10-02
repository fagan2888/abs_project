import pandas as pd 
import numpy as np 
import dateutil.relativedelta
import clean_data as c
pd.options.mode.chained_assignment = None  # default='warn'


if __name__ == '__main__':
	df_orig = pd.read_csv('freddie_origination_head_sample_2.csv')
	df_perf = pd.read_csv('freddie_performance_head_sample_2.csv')

	### Working of Performance File ################
	df_merged = c.Merge_Orig_Perf(df_orig, df_perf)

	#df_merged = pd.read_csv('freddie_merged.csv')
	df_merged.fillna('nan', inplace=True)

	df_merged, df_orig = c.Remove_Missing_Principal_Bonds(df_merged, df_orig)

	cols = list(df_merged.columns)
	df_mtx = np.array(df_merged)

	df_mtx, cols_added = c.Create_Pay_Cols(df_mtx)
	all_cols = cols + cols_added	# cols_added is 'prepay paid', 'interest paid', etc..
	df = pd.DataFrame(df_mtx, columns=all_cols)
	###############################################

	### Working on original data ##################
	df_orig = c.Calc_Prepay_Percent(df, df_orig)

	# divide by 100 because 'original_ltv' percent, not decimal
	df_orig['Home Price'] = df_orig['original_upb'] / (df_orig['original_ltv'] / 100.0)
	# Change the format of ltv, dummy variable
	df_orig = c.Revise_format(df_orig)
	df_orig.to_csv('freddie_final_2.csv', index=False)

