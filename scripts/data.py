# In[0] setup

import pandas as pd
import numpy as np

from fredapi import Fred

import sys
sys.path.append('D:/Data Science/Time Series Project/configurations')
import config as cfg


fred = Fred(api_key=cfg.api_key)


# In[1] extract raw data

raw_data_df = pd.read_csv(cfg.data_link, index_col=0)[1:]

convert_dict = pd.read_csv(cfg.fred_code, header=None)
convert_dict.columns = ['code', 'name']
convert_dict = dict(zip(convert_dict['code'].str.upper(), convert_dict['name']))

#raw_data_df.columns.map(convert_dict)
new_col_list = []   
for code in raw_data_df.columns.values:
    try:
        new_col_list.append(fred.get_series_info(code)['title'])
        #info = [fred.get_series_info(code)['title'] for code in raw_data_df.columns.values]
    except:
        try:
            new_col_list.append(convert_dict[code.upper()])
        except:
            raw_data_df = raw_data_df.drop([code], axis=1)
            print(code)

raw_data_df.columns = new_col_list
del new_col_list

raw_data_df.to_csv('D:/Data Science/Time Series Project/saves/raw_data.csv')


# FIXME
'''
for idx, bool in enumerate(raw_data_df.columns.map(convert_dict).isna()):
    if bool == True:
        convert_dict[raw_data_df.columns[idx]] = fred.get_series_info(raw_data_df.columns[idx])['title']
raw_data_df.columns = raw_data_df.columns.map(convert_dict)'''




# %%
