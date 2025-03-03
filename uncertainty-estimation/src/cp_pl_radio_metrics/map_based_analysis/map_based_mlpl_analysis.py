#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 19:20:04 2024

@author: ubuntu
"""

import pickle, ipdb
import numpy as np
from crepes.extras_with_targ_strangeness import hinge, margin, binning
from sklearn.metrics import mean_squared_error, mean_absolute_error

import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

#analysis of ma_
#cps_mond_norm_std req_conf=95, eff_coverage=89, knn=70, bins=10 PImean=26.0 
with open('../../../data/map_based_mlpl/map_based_mlpl_analysis.pkl', 'rb') as f:
    result, df_test, feat_cols = pickle.load(f)
    
y_min=-np.inf
y_max =np.inf    

# this is cps_mond_norm_std, the best for map_based_mlpl
name = result.config

#work with 25% of the easiest:
df_test = df_test[df_test.sigmas < 0.25]


#extract mondrian bins if key present
bins_test = None
if result.cps_obj['cps'].mondrian:
    bin_thresholds = result.cps_obj['mond_bins']
    bins_test = binning(df_test["pred"].values, bins=bin_thresholds)
    #bins_test = binning(result.df_test.pred.values, bins=bin_thresholds)

#df_test = result.df_test
#df_test['sigmas'] = sigmas

if "res" in result.config:
    #cps_mond_norm_res
    #sigmas = sigmas_test_knn_res
    df_test['sigmas'] = result.cps_obj['difficulty_estimator'].apply(X=df_test[feat_cols])
else:
    #sigmas = sigmas_test_knn_std
    #ipdb.set_trace()
    df_test['sigmas'] = result.cps_obj['difficulty_estimator'].apply(X=df_test[feat_cols].values,y=df_test["pred"].values)
    
try:  
    intervals = result.cps_obj['cps'].predict(df_test.pred.values, sigmas=df_test['sigmas'].values,bins=bins_test,lower_percentiles=2.5,higher_percentiles=97.5,y_min=y_min, y_max=y_max)
except Exception as e:
    print(f"Error occurred: {e}")
    print("DataFrame details:")
    ipdb.set_trace()
    print(df_test.info())

df_res = pd.DataFrame()
# Calculate interval widths
interval_widths = np.abs(intervals[:, 1] - intervals[:, 0])
mean_widths = np.mean(interval_widths)
std_widths = np.std(interval_widths)

lower = intervals[:,0]
upper = intervals[:,1]

df_res['int_widths'] = interval_widths
df_res['int_high'] = upper
df_res['int_low'] = lower
df_res['preds'] = df_test.pred.values
df_res['targets'] = df_test.targets.values
df_res['error'] = (df_res.targets - df_res.preds)

df_res['unscaled_asymmetry'] = abs(df_res['preds'] - df_res['int_high']) - abs(df_res['preds'] - df_res['int_low'])
most_asymmetric_interval = df_res.loc[df_res['unscaled_asymmetry'].idxmin()]

smallest_interval = df_res.loc[df_res['int_widths'].idxmin()]
largest_interval = df_res.loc[df_res['int_widths'].idxmax()]
median_interval = df_res['int_widths'].median()
#closest_row_to_median = df_res.loc[(df_res['int_widths'] - median_interval).abs().idxmin()]
#df_res[(df_res['preds'] < df_res['int_low']) | (df_res['preds'] > df_res['int_high'])]
