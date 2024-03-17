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
    plt.subplot(1,2,2)
    dataset[col_name].values_counts(normalize=True).plot.bar(rot=25)
    plt.ylabel(col_name)    
    plt.xlabel('% ' + 'distribution per category')
    plt.tight_layout()
    plt.show()

def num_visual(dataset, col_name):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    sns.kdeplot(dataset[col_name], color='g', shade=True)
    plt.subplot(1,2,2)
    sns.boxplot(dataset[col_name])
    plt.show()
    prt_stats(dataset, col_name)

def chk_missing(col): #FIXME
    if col.isin(['', None, pd.NaT, np.nan]) == True:
        pass

def imbalance_visual(dataset, tgt_feat=cfg.target):
    class_split = dataset[['id',tgt_feat]].groupby([tgt_feat]).count()
    pass



































'''------------------------------------------ETL functions------------------------------------------'''
from sklearn.preprocessing import FunctionTransformer
class CustomFunctionTransformer(FunctionTransformer):
    def __init__(self, func, validate=False, kw_args=None):
        self.kw_args = kw_args
        super().__init__(func=func, validate=validate)

    def fit(self, X, y=None):
        return super().fit(X, y, **self.kw_args) if self.kw_args else super().fit(X, y)

    def transform(self, X):
        return super().fit(X, **self.kw_args) if self.kw_args else super().fit(X)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return [f'{input_features}']
        #return super().get_feature_names_out(input_features)


















'''------------------------------------------ML functions------------------------------------------'''
# Result evaluation functions
def param_combinations(CV_model, mdl_param_grid:dict, param_to_chk:list):
    means = CV_model.cv_results_['mean_test_score']
    stds = CV_model.cv_results_['std_test_score']
    params = CV_model.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print('%f (%f) with %r' % (mean, stdev, param))

    pass