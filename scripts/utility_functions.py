import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

import sys
import os
parent_dir = os.path.dirname(os.getcwd())
sys.path.append(os.path.join(parent_dir, 'configurations'))
import config as cfg

'''general utility functions'''

'''------------------------------------------EDA functions------------------------------------------'''
def prt_stats(dataset, col_name):
    print(np.mean(dataset[col_name])):
    print(np.median(dataset[col_name]))
    print(dataset[col_name].skew())
    print(dataset[col_name].kurtosis(), '\n')

def cat_visual(dataset, col_name):
    plt.figure(figsize=(14,5))
    plt.subplot(1,2,1)
    sns.countplot(data=dataset, y=col_name)
    
