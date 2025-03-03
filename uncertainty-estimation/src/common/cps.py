
import ipdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

from crepes import ConformalClassifier, ConformalRegressor, ConformalPredictiveSystem, __version__
from crepes.extras_with_targ_strangeness import hinge, margin, binning, DifficultyEstimator

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from crepes import WrapRegressor
from mapie.metrics import regression_coverage_score


print(f"crepes v. {__version__}")

def load_column_mapping(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config['column_mapping']

def rename_columns(test_data, mapping_file):
    # Load column mapping
    column_mapping = load_column_mapping(mapping_file)
    
    # Rename columns
    test_data = test_data.rename(columns=column_mapping)
    
    return test_data


def rmse(y, y_pred):
    return np.sqrt(np.mean(np.square(y - y_pred)))  

def get_metrics(y_true, y_pred, y_pis, n_bins=10):
    metrics = {}
    metrics["coverage_score"] = -1 
    metrics["mean_pred_int_width"] = -1
    metrics["local_coverage_score"] = -1 

    # replace inf values with a large number or some other placeholder
    #y_pis = np.where(np.isinf(y_pis), np.finfo(np.float64).max, y_pis)
    #y_true = np.where(np.isinf(y_true), np.finfo(np.float64).max, y_true)    
    if np.any(np.isinf(y_pis)):
        return metrics
    
    # global coverage score
    coverage_score = regression_coverage_score(y_true, y_pis[:, 0], y_pis[:, 1])
    metrics["coverage_score"] = round(coverage_score,2)

    # mean prediction interval
    pred_int_width = np.abs(y_pis[:, 1] - y_pis[:, 0])
    mean_pred_int_width = np.mean(pred_int_width)
    metrics["mean_pred_int_width"] = mean_pred_int_width

    # local coverage score
    bins = np.quantile(y_true, np.linspace(0, 1, n_bins + 1))
    bins[-1] += 1
    bin_ids = np.digitize(y_true, bins)
    coverage_score_list = []
    n_obs_list = []
    for bin_id in np.unique(bin_ids):
        coverage_score_list.append(
            regression_coverage_score(
                y_true[bin_ids == bin_id],
                y_pis[bin_ids == bin_id, 0],
                y_pis[bin_ids == bin_id, 1],
            )
        )
        n_obs_list.append(np.sum(bin_ids == bin_id))
    local_coverage_score = {
        f"[{bins[ii]},{bins[ii+1]})": {
            "n_obs": n_obs_list[ii],
            "coverage_score": coverage_score_list[ii],
            "mid_bin": (bins[ii] + bins[ii + 1]) / 2,
        }
        for ii in range(len(coverage_score_list))
    }
    metrics["local_coverage_score"] = local_coverage_score

    return metrics

def plot_metrics(metrics,title):
    

    # local coverages
    x = []
    y = []
    for k, v in metrics["local_coverage_score"].items():
        x.append(v["mid_bin"])
        y.append(v["coverage_score"])
    plt.plot(
        x,
        y,
        ":+",
        label="local",
    )
    plt.hlines(
        metrics["coverage_score"],min(x),max(x),
        label="global",
    )
    plt.title(
        title+f" coverage={metrics['coverage_score']:.0%}"
    )
    plt.grid()
    plt.xlabel("Pred Pathloss [dB]")
    plt.ylabel("Coverage score")
    plt.legend(title="Coverage score")
    plt.show()

def plot_pred_intervals(y_true, y_pred, intervals,title=''):
    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()
    # Prepare warnings when the prediction interval does not contain the true value
    
    warning1 = y_true > intervals[:,1]
    warning2 = y_true < intervals[:,0]
    warnings = warning1 + warning2
    pred_coverage = len(y_pred[~warnings])/len(y_pred)
    
    high_interval = intervals[:,1][~warnings] - y_pred[~warnings]
    low_interval = y_pred[~warnings] - intervals[:,0][~warnings]
    
    error = np.stack((high_interval, low_interval))
    num_bad_interval = len(error[error<0])
    error[error<0] = 0
    #does the prediciton-interval cover the true value
    notrue_coverage1 = y_true > intervals[:,1]
    notrue_coverage2 = y_true < intervals[:,0]
    notrue_coverage = notrue_coverage1 + notrue_coverage2
    coverage = len(y_true[~notrue_coverage])/len(y_true)

    
    width = abs(intervals[:,1] - intervals[:,0]).mean()
    width_std = abs(intervals[:,1] - intervals[:,0]).std()
    
    plt.plot(y_true, y_true, label="True values", color="black")
    inlier_rmse = rmse(y_true[~warnings],y_pred[~warnings])
    plt.errorbar(y_true[~warnings], y_pred[~warnings], error, alpha=0.5,fmt="o",color="blue",elinewidth=1,capsize=2,label="True Val In Prediction interval")
    
    plt.title(title+f"\nPrediction intervals with coverage {coverage:.4f} and width (mean,std): {width:.4f}, {width_std:.4f} \n num_bad_interval:{num_bad_interval:.0f} \nRMSE: {inlier_rmse:.4f}")
    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    
    plt.legend()
    plt.show()
    
    ##################
    #Plot intervals that do not contain the true value
    
    high_interval = intervals[:,1][warnings] - y_pred[warnings]
    low_interval = y_pred[warnings] - intervals[:,0][warnings]
    
    error = np.stack((high_interval, low_interval))
    num_bad_interval = len(error[error<0])
    error[error<0] = 0
    
    error[error<0]=0
    #true-val outside of interval (outlier) calculations
    width = (intervals[:,1][warnings] - intervals[:,0][warnings]).mean()
    width_std = (intervals[:,1][warnings] - intervals[:,0][warnings]).std()

    plt.plot(y_true, y_true, label="True values", color="black")
    outlier_rmse = rmse(y_true[warnings],y_pred[warnings])
    plt.errorbar(y_true[warnings], y_pred[warnings], error, alpha=0.5,fmt="o",color="red",elinewidth=1,capsize=2,label="Prediction interval ERROR")
    plt.title(title+f"\n True-Val in prediction intervals (coverage) {coverage:.4f} and width (mean,std): {width:.4f}, {width_std:.4f} \n Pred-Val in prediction intervals: {pred_coverage:.4f}\n num_bad_interval:{num_bad_interval:.0f}  \nRMSE: {outlier_rmse:.4f}")
    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    
    plt.legend()
    plt.show()
    
    inter_diff = abs(intervals[:,0]-intervals[:,1])

    plt.title(title)
    plt.scatter(x=y_true,y=inter_diff,marker="+")
    plt.ylabel('interval widths')
    plt.xlabel('True Pathloss [dB]')
    plt.show()
    
    return

def plot_xcorr(df):
    import seaborn as sns
    df['err']=abs(df.pred-df.targets)
    sns.heatmap(df.corr(), annot=True,annot_kws={"size": 6})
   
    plt.show()
  
def diff_intervals(cr_norm, y_pred, sigmas,conf,y_min,y_max):
    intervals_cr_norm = cr_norm.predict(y_pred.squeeze(),sigmas.squeeze(),confidence=conf,y_min=y_min, y_max=y_max)
    return intervals_cr_norm


def cps_regression(seed, df_train, df_cal, df_test,feat_cols, knn_num,bin_num,targ_strangeness, req_conf_intervals, include_pred_metadata=False):
    
    #do not limit range given for prediction intervals

    ##############################################
    #first pass is 90%, then conformal predictive systems does 90% confidence intervals for plots
    #from which the seed mean results are calculated

    confidence = 0.9 
    y_min=-np.inf
    y_max=np.inf
    k_num_nn=knn_num
    mond_bins = bin_num
    ###############################################
    
    
    k_num_nn = knn_num
    
    plot_debug =False
    #prepare results dataframe
    df_cfg_results = pd.DataFrame(columns = \
                         ['seed','config','req_conf','eff_coverage',\
                         'knn','bins','PI_mean','PI_std','cps_obj','sigmas','bins_test','df_test'])
    
    residuals_cal = df_cal.targets.values - df_cal.pred.values 
    knn_targ_strange_flag = targ_strangeness

    
    ###############################################
    #difficulty estimator : knn_std or knn_target_strangeness
    #using the default number of nearest neighbors
    de_knn_std = DifficultyEstimator(knn_target_strangeness=knn_targ_strange_flag)
    de_knn_std.fit(X=df_train[feat_cols].values, y= df_train["targets"].values,\
                   k=k_num_nn,scaler=True)
    
    sigmas_cal_knn_std = de_knn_std.apply(X=df_cal[feat_cols].values, y=df_cal["targets"].values)
    cr_norm_knn_std = ConformalRegressor()

    cr_norm_knn_std.fit(residuals_cal.squeeze(), sigmas=sigmas_cal_knn_std)
    
    sigmas_test_knn_std = de_knn_std.apply(X=df_test[feat_cols].values,y=df_test["pred"].values)
    intervals = diff_intervals(cr_norm_knn_std,df_test["pred"].values, sigmas_test_knn_std,confidence,y_min,y_max)
    
    
    if plot_debug:
        my_metrics = get_metrics(df_test["targets"].values, df_test["pred"].values, intervals, n_bins=10)
        plot_metrics(my_metrics,"de knn_std")
        plot_pred_intervals(df_test["targets"].values ,df_test["pred"].values , intervals, title='KNNstd confidence ='+str(confidence)+' min/max: '+str(y_min)+'/'+str(y_max))
    
    
    if not knn_targ_strange_flag:
        ###############################################
        #difficulty estimator : KNN_res
        residuals = (df_train["pred"].values - df_train["targets"].values)
        de_knn_res = DifficultyEstimator(knn_target_strangeness=knn_targ_strange_flag)
        
        de_knn_res.fit(X=df_train[feat_cols].values, residuals=residuals, scaler=True)
        sigmas_cal_knn_res = de_knn_res.apply(X=df_cal[feat_cols])
        
        cr_norm_knn_res = ConformalRegressor()
        cr_norm_knn_res.fit(residuals_cal.squeeze(), sigmas=sigmas_cal_knn_res)
        sigmas_test_knn_res = de_knn_res.apply(X=df_test[feat_cols])
        
        intervals = diff_intervals(cr_norm_knn_res,df_test["pred"].values, sigmas_test_knn_res,confidence,y_min,y_max)
               
        
        if plot_debug:
            my_metrics = get_metrics(df_test["targets"].values, df_test["pred"].values, intervals, n_bins=10)
            plot_metrics(my_metrics,"de knn_res")
            plot_pred_intervals(df_test["targets"].values ,df_test["pred"].values , intervals, title='KNNres confidence ='+str(confidence)+' min/max: '+str(y_min)+'/'+str(y_max))
        
        
    
        #######
        #Use Mondrian conformal regressors with model, using with de_std
        #interval percentile
        
        bins_cal, bin_thresholds = binning(sigmas_cal_knn_std, bins=mond_bins)
        
        cr_mond = ConformalRegressor()
        
        cr_mond.fit(residuals_cal, bins=bins_cal)
        bins_test = binning(sigmas_test_knn_res, bins=bin_thresholds)
        
        intervals_mond = cr_mond.predict(df_test["pred"].values, bins=bins_test,confidence=confidence,y_min=y_min,y_max=y_max)
        
        if plot_debug:
            my_metrics = get_metrics(df_test["targets"].values, df_test["pred"].values, intervals_mond, n_bins=10)
            plot_metrics(my_metrics,"mondrian")
            plot_pred_intervals(df_test["targets"].values ,df_test["pred"].values , intervals_mond, title='Mondrian confidence ='+str(confidence)+' min/max: '+str(y_min)+'/'+str(y_max))
            
            inter_diff = abs(intervals_mond[:,0]-intervals_mond[:,1])
            plt.title('mondrian intervals: binning sigmas_test_knn_std')
            plt.scatter(x=df_test["targets"],y=inter_diff,marker="+")
            plt.ylabel('interval widths')
            plt.xlabel('True Pathloss [dB]')
            plt.show()
        
    
    ####################
    #Conformal Predictive Systems (predict the cdf)
    #cps_norm = ConformalPredictiveSystem().fit(residuals_cal,
    #                                       sigmas=sigmas_cal_knn_std)
    
    cps_bins = mond_bins 
    bins_cal, bin_thresholds = binning(df_cal.pred.values, bins=cps_bins)
    
    cps_norm_std = ConformalPredictiveSystem().fit(residuals_cal,
                                                    sigmas=sigmas_cal_knn_std)
    
    cps_mond_norm_std = ConformalPredictiveSystem().fit(residuals_cal,
                                                    sigmas=sigmas_cal_knn_std,
                                                    bins=bins_cal)
    
    if not knn_targ_strange_flag:
        cps_mond_norm_res = ConformalPredictiveSystem().fit(residuals_cal,
                                                    sigmas=sigmas_cal_knn_res,
                                                    bins=bins_cal)
    
    bins_test = binning(df_test.pred.values, bins=bin_thresholds)
    
    if knn_targ_strange_flag:
        all_cps = {
            "cps_norm_targ_strng": {
                "cps": cps_norm_std
            },
            "cps_mond_norm_targ_strng": {
                "cps": cps_mond_norm_std
            }
        }
    else:
        all_cps = {
            "cps_norm_std": {
                "cps": cps_norm_std
            },
            "cps_mond_norm_std": {
                "cps": cps_mond_norm_std
            },
            "cps_mond_norm_res": {
                "cps": cps_mond_norm_res
            }
        }
    
    
    if plot_debug:
        for idx, name in enumerate(all_cps.keys()):
            if "res" in name:
                #cps_mond_norm_res
                sigmas = sigmas_test_knn_res
            else:
                sigmas = sigmas_test_knn_std
            
            p_values = all_cps[name].predict(df_test.pred.values,
                                         sigmas=sigmas,
                                         bins=bins_test,
                                         y=df_test.targets.values)
            
            plt.scatter(np.sort(p_values),
                            [(i+1)/len(df_test.targets.values) for i in range(len(df_test.targets.values))],
                            label=name, c="y", marker=".", alpha=0.1)
            
            plt.plot([0,1],[0,1],"r--")
            plt.legend()
            plt.ylabel("fraction")
            plt.xlabel("p value")
            plt.show()
    

    def collect_results(intervals, eff_coverage,title, conf_req,knn, bins, cps_obj,sigmas, bins_test):
        #do not collect results that had inf values
        if eff_coverage == -1:
            return
        title_full = title+' '+str(conf_req)+'%'
        interval_widths = intervals[:,1]-intervals[:,0]
        inter_mean = round(interval_widths.mean(),2)
        inter_std = round(interval_widths.std(),2)
        print(title_full+ ' coverage: '+str(eff_coverage*100)+'% interval_widths mean: '+str(inter_mean)+' std:'+str(inter_std))
       
        df_cfg_results.loc[len(df_cfg_results)] = [seed,title,conf_req,eff_coverage*100,knn,bins,inter_mean,inter_std, cps_obj,sigmas, bins_test,df_test[['targets', 'pred']]]
        
        if plot_debug:
            inter_diff = interval_widths
            plt.scatter(x=df_test["targets"],y=inter_diff,marker="+")
            plt.title(title)
            plt.ylabel('interval widths')
            plt.xlabel('True Pathloss [dB]')
            plt.show()
        
        return
    
    def calculate_percentiles(confidence):
        """Calculate lower and upper percentiles based on confidence interval."""
        lower_percentile = (100 - confidence) / 2
        higher_percentile = 100 - lower_percentile
        return lower_percentile, higher_percentile
    
    for idx, name in enumerate(all_cps.keys()):
        if "res" in name:
            sigmas = sigmas_test_knn_res
        else:
            sigmas = sigmas_test_knn_std  

        if include_pred_metadata:
            all_cps[name]["difficulty_estimator"]= de_knn_std
            if 'mond' in name:
                all_cps[name]["mond_bins"]=bin_thresholds
           
        #99% conf interval
        if 99 in req_conf_intervals:
            intervals_99 = all_cps[name]["cps"].predict(df_test.pred.values, sigmas=sigmas,bins=bins_test,lower_percentiles=1,higher_percentiles=100)
            #intervals_99 = all_cps[name].predict(df_test.pred.values, sigmas=sigmas,bins=bins_test,lower_percentiles=1,higher_percentiles=100)
            my_metrics = get_metrics(df_test["targets"].values, df_test["pred"].values, intervals_99, n_bins=10)
            intervals_99_actual = my_metrics['coverage_score']
    
            
        #95% confident
        if 95 in req_conf_intervals:
            intervals_95 = all_cps[name]["cps"].predict(df_test.pred.values, sigmas=sigmas,bins=bins_test,lower_percentiles=2.5,higher_percentiles=97.5)                   
            #intervals_95 = all_cps[name].predict(df_test.pred.values, sigmas=sigmas,bins=bins_test,lower_percentiles=2.5,higher_percentiles=97.5)                       
            my_metrics = get_metrics(df_test["targets"].values, df_test["pred"].values, intervals_95, n_bins=10)
            intervals_95_actual = my_metrics['coverage_score']
        
        #90% Confidence interval
        if 90 in req_conf_intervals:
            intervals_90 = all_cps[name]["cps"].predict(df_test.pred.values, sigmas=sigmas,bins=bins_test,lower_percentiles=5,higher_percentiles=95)                           
            #intervals_90 = all_cps[name].predict(df_test.pred.values, sigmas=sigmas,bins=bins_test,lower_percentiles=5,higher_percentiles=95)                           
            my_metrics = get_metrics(df_test["targets"].values, df_test["pred"].values, intervals_90, n_bins=10)
            intervals_90_actual = my_metrics['coverage_score']
      

        #concat results all together 
        if 99 in req_conf_intervals:
            collect_results(intervals_99, intervals_99_actual, name, 99, k_num_nn, mond_bins, all_cps[name], sigmas, bins_test)
        if 95 in req_conf_intervals:
            collect_results(intervals_95, intervals_95_actual, name, 95, k_num_nn, mond_bins, all_cps[name], sigmas, bins_test)
        if 90 in req_conf_intervals:
            collect_results(intervals_90, intervals_90_actual, name, 90, k_num_nn, mond_bins, all_cps[name], sigmas, bins_test )
       

        
    return df_cfg_results
