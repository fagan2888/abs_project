import pandas as pd 
import numpy as np 

df_orig = pd.read_csv('freddie_origination_sample.csv')

# 'original_upb' = unpiad principal balance at note date
# divide by 100 because 'original_ltv' percent, not decimal
df_orig['Home Price'] = df_orig['original_upb'] / (df_orig['original_ltv'] / 100.0)

