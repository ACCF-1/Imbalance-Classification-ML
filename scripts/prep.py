# In[0] Libraries & Setup
'''Setup'''
import pandas as pd
import numpy as np
import utility_functions as uf

import sys
import os
parent_dir = os.path.dirname(os.getcwd())
sys.path.append(os.path.join(parent_dir, 'configurations'))
import config as cfg

import warnings
warnings.filterwarnings("ignore")

import data

raw_data_df = data.raw_data_df.copy()

tgt_convert = {'A':1, 'B':2}


# In[1] data overview
'''Overview of data'''

raw_overview = {'shape': raw_data_df.shape,
                'top rows': raw_data_df.head(),
                'bot rows': raw_data_df.tail(),
                'type': raw_data_df.dtypes,
                'info': raw_data_df.describe(include='all')}

for info in raw_overview.items():
    print(info)

uf.imbalance_visual(raw_data_df)

print('duplicated values:\n', raw_data_df.duplicated().any(), '\n')
print('unique values:\n', raw_data_df.nunique(), '\n')

print('null values:\n', raw_data_df.isna().sum(), '\n')

# In[2] Global cleansing
'''Global cleansing'''

