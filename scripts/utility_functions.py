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

import inspect

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
    class_split.index = class_split.index.astype(str)
    one_class_cnt = class_split.loc['1', 'id']
    zero_class_cnt = class_split.loc['0', 'id']
    print('major class: ', zero_class_cnt, '\n',
          'minority class: ', one_class_cnt, '\n',
          'imbalance ratio: ', round(zero_class_cnt/one_class_cnt, 4), '\n')
    plt.bar(class_split.index, class_split['id'])
    plt.show()

def detect_outlier(dataset:pd.DataFrame, tgt_col, feats:list, threshold:int =2):
    df = dataset.copy()
    df[cfg.target] = tgt_col
    fig, axs = plt.subplots(len(feats), 1, figsize=(8, 4*len(feats)))
    for idx, feat in enumerate(feats):
        if np.issubdtype(df[feat].dtypes, np.number) == True:
            sns.boxplot(x=df[feat], color='darkblue', ax=axs[idx])
            axs[idx].set_title('Boxplot - ' + df[feat].name)
    fig.tight_layout()
    fig.show()
    fig.savefig(cfg.saves_path + '/' + 'outlier chk - boxplot' + '.pdf')

def feat_vs_tgt_chk(df:pd.DataFrame, tgt_col, feats:list, threshold:float, status:str, graph_type:str):
    dataset = df.copy()
    dataset[cfg.target] = tgt_col.copy()
    fig, axs = plt.subplots(len(feats), 1, figsize=(8, 4*len(feats)))
    for idx, feat in enumerate(feats):
        graph_data = dataset[[feat, cfg.target]].groupby([feat]).mean().sort_values(by=feat, ascending=False).reset_index()
        graph_data['Abnormal'] = graph_data[cfg.target] > threshold
        try:
            if graph_type == 'barplot':
                sns.barplot(data=graph_data, x=feat, y=cfg.target, width=0.7, hue='Abnormal',
                            palette={True:'red', False:'navy'}, ax=axs[idx])
                axs[idx].set_title(feat)
                axs[idx].set_xticklabels(graph_data[feat].sort_values(ascending=True), rotation=90)
            elif graph_type == 'pointplot':
                sns.pointplot(data=graph_data, x=feat, y=cfg.target, palette='deep', ax=axs[idx])
                axs[idx].set_title(feat)
                axs[idx].set_xticklabels(graph_data[feat].sort_valuues(ascending=True), rotation=90)
        except:
            print(dataset[[feat, cfg.target]].groupby([feat]).mean().sort_values(by=feat, ascending=False))
            return
    fig.tight_layout()
    fig.show()
    fig.savefig(cfg.saves_path + '/' + inspect.stack()[0][3] + ' - ' + status + ', ' + graph_type + '.pdf')

def non_norm_identifier(dataset:pd.DataFrame) ->list:
    non_norm_cols = []
    other_cols = set()
    binary_cat_cols = []
    try:
        for col in dataset.columns:
            if dataset[col].min() < 0 or dataset[col].max() > 1:
                non_norm_cols.append(col)
    except:
        other_cols.add(col)

    try:
        for col in dataset.columns:
            if all(dataset[col].unique()==[0,1])==True or all(dataset[col].unique()==0) or all(dataset[col].unique()==1):
                non_norm_cols.append(col)
    except:
        other_cols.add(col)
    return non_norm_cols, binary_cat_cols, list(other_cols)


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

def split_num_and_cat(dataset:pd.DataFrame, to_return:int):
    non_norm_cols, binary_cat_cols, _ = non_norm_identifier(dataset)

    cat_in_non_norm_cols = [col for col in non_norm_cols if "|" not in col and dataset[col].dtype == int]
    num_in_non_norm_cols = [col for col in non_norm_cols if col not in cat_in_non_norm_cols]

    cat_cols = binary_cat_cols + cat_in_non_norm_cols
    num_cols = [col for col in dataset.columns if col not in cat_cols]

    if to_return == 1:
        return cat_in_non_norm_cols
    elif to_return == 2:
        return num_in_non_norm_cols
    elif to_return == 3:
        return cat_cols
    elif to_return == 4:
        return [col for col in num_cols if col not in num_in_non_norm_cols]
    elif to_return == 5:
        return binary_cat_cols
    elif to_return == 6:
        return non_norm_cols

def get_idx_for_col_trans(feat_names, col_type:str):
    feat_idx_mapping = []
    for i, feat_name in enumerate(feat_names):
        if col_type == 'num':
            if 'scaling__' in feat_name or 'age trans__' in feat_name:
                feat_idx_mapping.append(i)
        elif col_type == 'cat':
            if 'cat trans__' in feat_name or 'remainder__' in feat_name:
                feat_idx_mapping.append(i)
    return feat_idx_mapping

def get_feature_correlation(df:pd.DataFrame, top_n:int=None, 
                            corr_method:str='spearman', remove_duplicates:bool=True, 
                            remove_self_correlations:bool=True)->pd.DataFrame:
    '''
    Compute the feature correlation and sort feature pairs based on their correlation

    :param df: Dataframe with predictor variables
    :param top_n: Top N feature pairs to be reported (if None, return all pairs)
    :param corr_method: Correlation computation method
    :param remove_duplicates: whether duplicate features must be removed
    :param remove_self_correlations: whether self correlation will be removed

    :return: DataFrame
    '''
    corr_matrix_abs = df.corr(method=corr_method).abs
    corr_matrix_abs_us = corr_matrix_abs.unstack()
    sorted_corr_feats = corr_matrix_abs_us.sort_values(kind='quicksort', ascending=False).reset_index()

    if remove_self_correlations:
        sorted_corr_feats = sorted_corr_feats[(sorted_corr_feats.level_0!=sorted_corr_feats.level_1)]

    if remove_duplicates:
        sorted_corr_feats = sorted_corr_feats.iloc[:-2:2]

    sorted_corr_feats.columns = ['Feature 1', 'Feature 2', 'Correlation (abs)']
    sorted_corr_feats = sorted_corr_feats[~sorted_corr_feats.apply(frozenset, axis=1).duplicated()]
    if top_n:
        return sorted_corr_feats[:top_n]

    return sorted_corr_feats

from sklearn.base import BaseEstimator, TransformerMixin

class CorrFilterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._columns = None

    @property
    def columns_(self):
        if self._columns is None:
            raise Exception('CorrFilterTransformer has not been fitted yet')
        return self._columns

    def fit(self, X, y=None):
        corr_df = get_feature_correlation(X)
        high_corr_cols = corr_df[corr_df['Correlation (abs)'] > 0.9]['Feature 1'].unique()
        self._columns = high_corr_cols
        return self

    def transform(self, X, y=None):
        X = X[self._columns]
        return X

    def get_feature_names_out(self, input_features=None):
        return self._columns    
    

'''------------------------------------------ML functions------------------------------------------'''
# Result evaluation functions
def param_combinations(CV_model, mdl_param_grid:dict, param_to_chk:list):
    means = CV_model.cv_results_['mean_test_score']
    stds = CV_model.cv_results_['std_test_score']
    params = CV_model.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print('%f (%f) with %r' % (mean, stdev, param))

    scores = np.array(means).reshape(len(mdl_param_grid[param_to_chk[0]]), len(mdl_param_grid[param_to_chk[1]]))
    for i, val in enumerate(mdl_param_grid[param_to_chk[0]]):
        plt.plot(mdl_param_grid[param_to_chk[1]], scores[i], label=param_to_chk[0] + ': ' + str(val))
    plt.legend()
    plt.xlabel('n_estimators')
    plt.ylabel('F2 score')
    plt.savefig(cfg.saves_path + '/' + param_to_chk[0] + '_vs_' + param_to_chk[1] + '.png')