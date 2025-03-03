import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import ipdb, glob
from mapie.metrics import regression_coverage_score
import argparse, os, json
from crepes.extras_with_targ_strangeness import hinge, margin, binning

from sklearn.metrics import mean_squared_error, mean_absolute_error

#for gen_all_plts debugging
#set the request confidence interval
#request 95%  but the actual will be lower
higher_percentile = 97.5
lower_percentile = 100-higher_percentile

def get_metrics(y_true, y_pred, y_pis, n_bins=10):
    metrics = {}

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

def plot_metrics(metrics,title, pred_type_units):
    

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
    plt.xlabel("Prediction "+pred_type_units)
    plt.ylabel("Coverage score")
    plt.legend(title="Coverage score")
    plt.show()

def plot_cpd(cpds,y_hat_full,test_index,df_test): 
    cpd = cpds[test_index]
    
    p = np.array([i/len(cpd) for i in range(len(cpd))])
    
    lower_index = np.where(p<=lower_percentile/100)[0][-1]
    mid_index = np.where(p>=0.50)[0][0]
    upper_index = np.where(p>=higher_percentile/100)[0][0]
    
    low_percentile = cpd[lower_index]
    median = cpd[mid_index]
    high_percentile = cpd[upper_index]
    
    plt.figure(figsize=(6,6))
    plt.plot([y_hat_full[test_index],y_hat_full[test_index]],[0,1], color="tab:orange")
    plt.plot([df_test["targets"].values[test_index],df_test["targets"].values[test_index]],[0,1], color="tab:red")
    plt.xlabel("y")
    plt.ylabel("Q(y)")
    plt.ylim(0,1)
    
    plt.plot([median,median],[0,1],"g--")
    plt.plot([low_percentile,low_percentile],[0,1],"y--")
    
    low_tile = lower_percentile/100
    high_tile = higher_percentile/100
    
    plt.legend(["ŷ","target","$y_{0.5}$","[$y_{low_tile}$,$y_{high_tile}$]"])
    plt.plot([high_percentile,high_percentile],[0,1],"y--")
    plt.plot(cpd,p, color="tab:blue")
    rectangle = plt.Rectangle((low_percentile,0),
                              abs(high_percentile-low_percentile),1, color="y",
                              alpha=0.05)
    plt.gca().add_patch(rectangle)
    plt.show()

def cps_result_plots(all_cps, pred_type_units, gen_all_plts, results_path, df_test_all, feat_cols, rmse_quant_flag):
  y_min=-np.inf
  y_max =np.inf
    
  if gen_all_plts:  

    for idx, result in all_cps.iterrows() :   

        sigmas =result.sigmas
        name = result.config
        bins_test =result.bins_test
        df_test = df_test_all
        cpds = result.cps_obj['cps'].predict(df_test.pred.values,
                                 sigmas=sigmas,
                                 bins=bins_test,y_min=y_min, y_max=y_max,
                                 return_cpds=True)
        
        print(f"No. of test instances: {len(df_test.pred.values)}")
        print(f"Shape of cpds: {cpds.shape}")
        
        
        #intervals = result.cps_obj.predict(df_test.pred.values, sigmas=sigmas,bins=bins_test,lower_percentiles=5,higher_percentiles=95,y_min=y_min, y_max=y_max)                             

        intervals = result.cps_obj['cps'].predict(df_test.pred.values, sigmas=sigmas,bins=bins_test,lower_percentiles=5,higher_percentiles=95,y_min=y_min, y_max=y_max)                                     
        lower = intervals[:,0]
        upper = intervals[:,1]
        
        inter_diff = abs(lower-upper)
        plt.scatter(x=df_test["targets"],y=inter_diff,marker="+")
        plt.title(name+': percentile widths between: '+str(higher_percentile)+'% to '+str(lower_percentile)+'%'\
                      +' mean: '+str(inter_diff.mean().round(2))+' std: '+str(inter_diff.std().round(2)))
        plt.ylabel('interval widths')
        plt.xlabel('True '+pred_type_units)
        plt.show()
        
        intervals = np.stack([lower,upper]).T
        my_metrics = get_metrics(df_test["targets"].values, df_test["pred"].values, intervals, n_bins=10)
        plot_metrics(my_metrics,"Conformal Pred Systems", pred_type_units)
        
        y_hat_full = df_test.pred.values
        test_index = np.random.randint(len(y_hat_full)) # A test object is randomly selected
        plot_cpd(cpds,y_hat_full,test_index,df_test) 
    
    
    colors = ["b","r","g","k","m"]

    for idx, result in all_cps.iterrows() :    
        sigmas =result.sigmas
        name = result.config
        bins_test =result.bins_test
        #df_test = result.df_test
        
        #######################################
        intervals = result.cps_obj['cps'].predict(df_test.pred.values, sigmas=sigmas,bins=bins_test,lower_percentiles=5,higher_percentiles=95,y_min=y_min, y_max=y_max)                             
    
        lower = intervals[:,0]
        upper = intervals[:,1]
        cps_interval_sizes = upper - lower
        plt.ylabel("CDF")
        plt.xlabel("Interval sizes")
        plt.title(name+': CDF vs Interval size')
        
        data = [i/len(cps_interval_sizes) for i in range(1,len(cps_interval_sizes)+1)]
        plt.plot(np.sort(cps_interval_sizes),data,linestyle="solid", c=colors[idx],label=name)
        
    plt.legend()
    plt.show()
    

  #start main loop 
  for idx, result in all_cps.iterrows() : 

        alpha=0.2
        name = result.config

        if ('map' in results_path):
            #keeping test the same size, geographic splitting was not done
            df_test = df_test_all.sample(frac=0.5, random_state=result.seed).reset_index(drop=True)
        else:
            df_test = df_test_all

        #extract mondrian bins if key present
        bins_test = None
        if result.cps_obj['cps'].mondrian:
            #ipdb.set_trace()
            bin_thresholds = result.cps_obj['mond_bins']
            bins_test = binning(df_test["pred"].values, bins=bin_thresholds)
            #bins_test = binning(result.df_test.pred.values, bins=bin_thresholds)
        
        if "res" in result.config:
            df_test['sigmas'] = result.cps_obj['difficulty_estimator'].apply(X=df_test[feat_cols])
        else:
            df_test['sigmas'] = result.cps_obj['difficulty_estimator'].apply(X=df_test[feat_cols].values,y=df_test["pred"].values)
            
        low_percentile = (100 - result.req_conf) / 2
        high_percentile = 100 - low_percentile    

        try:  
            intervals = result.cps_obj['cps'].predict(df_test.pred.values, sigmas=df_test['sigmas'].values,bins=bins_test,lower_percentiles=low_percentile,higher_percentiles=high_percentile,y_min=y_min, y_max=y_max)
        except Exception as e:
            print(f"Error occurred: {e}")
            print("DataFrame details:")
            print(df_test.info())
            
        
        rmse_quantiles = {0.25: None, 0.50: None, 0.75: None}  # Initialize dictionary
        if rmse_quant_flag: 
            sigmas_thresholds = np.quantile(df_test.sigmas.values, list(rmse_quantiles.keys()))
            for quantile, threshold in zip(rmse_quantiles.keys(), sigmas_thresholds):
                df_filt = df_test[df_test.sigmas < threshold]
                rmse_quantiles[quantile] = round(mean_squared_error(df_filt['targets'].values, df_filt.pred.values)**0.5, 2)
        
        
        orig_rmse = round(mean_squared_error(df_test['targets'].values,df_test['pred'].values)**0.5,2)
        
        rsme_str = 'RMSE over least difficult [25%, 50%, 75%,  100%]: '+str(rmse_quantiles[0.25])+', ' +str(rmse_quantiles[0.50])+', '+str(rmse_quantiles[0.75])+', '+str(orig_rmse)
        
        #added to get the new metrics based on model predicting different data
        my_metrics = get_metrics(df_test["targets"].values, df_test["pred"].values, intervals, n_bins=10)
        
        lower = intervals[:,0]
        upper = intervals[:,1]
        indexes= np.argsort(df_test.pred.values) 
        
        # Calculate interval widths
        interval_widths = np.abs(upper - lower)
        
        # Create a figure with a specific layout
        plt.figure(figsize=(12, 8))
        #uncomment for larger font
        #plt.rcParams['font.family'] = 'Times New Roman'
        #plt.rcParams['font.size'] = 20 # Default font size
        # Create a gridspec layout
        gs = plt.GridSpec(1, 2, width_ratios=[3, 0.5])  # Reduced second column width
        
        # Main scatter plot with prediction intervals
        plt.subplot(gs[0])
        #plt.title('Prediction Intervals (PI) with '+str(my_metrics['coverage_score']*100)+'% Effective Coverage \n for '+name)
        plt.title('Prediction Intervals (PI) \n Method: '+name + '\n'+'Effective Coverage: '+str(my_metrics['coverage_score']*100)+'%')
        plt.plot(df_test.pred.values[indexes], lower[indexes], 
                 color="r", alpha=alpha, label='Prediction Intervals')
        plt.plot(df_test.pred.values[indexes], upper[indexes], 
                 color="r", alpha=alpha)
        plt.scatter(df_test.pred.values[indexes],df_test.targets.values[indexes],
                   color="b", marker="o", alpha=alpha, label='Predictions')
        plt.scatter(df_test.pred.values[indexes],df_test.pred.values[indexes],
                    color="y", marker=".", alpha=alpha, label='Reference')
        plt.xlabel("Predicted "+pred_type_units + ('\n'+rsme_str if rmse_quant_flag else ''))
        plt.ylabel("Target "+pred_type_units)
        plt.legend(loc='best', fontsize=8)
        
        # Vertical boxplot of interval widths
        plt.subplot(gs[1])
        sns.boxplot(y=interval_widths, color=(1, 0.6, 0.6, alpha))
        plt.title('Interval Widths'+'\n Mean Width='+str(round(my_metrics['mean_pred_int_width'],2)))
        plt.ylabel('Width [dB]')
        plt.xticks([])  # Remove x-axis ticks
        plt.ylim(bottom=0)  # Ensure y-axis starts at 0
        plt.tight_layout()

        if rmse_quant_flag: 
            filename = name + '='+str(my_metrics['coverage_score'])+'_conf_mean_'+str(round(my_metrics['mean_pred_int_width'],2))+' rmse:'+str(orig_rmse)
        else:
            filename = name + '='+str(my_metrics['coverage_score'])+'_conf_mean_'+str(round(my_metrics['mean_pred_int_width'],2))
        plt.savefig(results_path+'/'+filename+'.png', dpi=300, bbox_inches='tight')
        #plt.show()
        plt.clf()
    
def remove_trailing_slash(s):
    if s.endswith('/'):
        return s[:-1]
    return s    

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


if __name__ == "__main__":
   '''
   example how to run
   /cp_mm_targ_strg/src$ python cps_result_plots.py --results_file_path '../results/mlpl/ext_num/05-14_12h37m29s/all_cfg_results.pkl' --pred_type_units 'Path Loss [dB]'
   or use
   sh /cp_mm_targ_strg/src/gen_all_pis_plots.sh
   '''
    
   parser = argparse.ArgumentParser(description="Conformal Predictive Systems plot results based on path.")
   parser.add_argument("--results_file_path", type=str, help="Path to the pkl results file.")
   parser.add_argument("--pred_type_units", type=str, help="label for plots")
   parser.add_argument("--test_data_dir", type=str, help="dir for test data")
   parser.add_argument("--test_data_fname", type=str, help="test data filename")
   parser.add_argument("--gen_all_plts", action="store_true", help="Generate all debug plots if this flag is set")
   parser.add_argument("--rmse_quant_flag", action="store_true", help="Calc the RMSE of the cumulative quantiles")
   # Parse command-line arguments
   args = parser.parse_args()
   
   results_file_path = remove_trailing_slash(args.results_file_path)
   results_path = os.path.dirname(results_file_path) + '/'
   df_cfg_results_all = pd.read_pickle(results_file_path)
   df_cfg_results_all = df_cfg_results_all.reset_index(drop=True)
    
   min_PI_mean_idx = df_cfg_results_all.groupby('config')['PI_mean'].idxmin()
   #find knn and Mondrian bins for the lowest PI_mean per config
   all_cps = df_cfg_results_all.loc[min_PI_mean_idx]
   all_cps = all_cps.reset_index(drop=True)


   test_data_dir = args.test_data_dir
   
   test_data = pd.read_csv(test_data_dir+'/'+args.test_data_fname).dropna()
   
   
   if not ('mlpl' in test_data_dir or 'multimodal' in test_data_dir):
       N=100
       test_data['rsrp_est'] = test_data.apply(lambda row: row['tx_power'] - row['fspl_mlpl_fcn'] - 10 * np.log10(12 * N), axis=1)
       
       #drop fspl_mlpl_fcn
       test_data.drop(columns=['fspl_mlpl_fcn'], inplace=True)
      
   if 'multimodal' in test_data_dir:
       file_path = results_file_path .replace('all_cfg_results.pkl', 'feat_columns.csv')
       #feat_cols = pd.read_csv(file_path).columns.to_list()
       feat_cols = json.load(open(file_path))#.to_list()
       test_data = test_data.rename(columns={'labels': 'targets','preds': 'pred'})
   else:
       test_data = rename_columns(test_data, test_data_dir+'column_mapping.json')
       feat_cols =  pd.read_csv(test_data_dir+'feat_columns.csv').columns.to_list()

   all_cps = all_cps.reset_index(drop=True)
   cps_result_plots(all_cps, args.pred_type_units, args.gen_all_plts, results_path, test_data, feat_cols, args.rmse_quant_flag)   
   
   '''
   #redo with with eff_coverage greater than 89% 
   df_cfg_results_all = df_cfg_results_all[df_cfg_results_all.eff_coverage > 89]
   min_PI_mean_idx = df_cfg_results_all.groupby('config')['PI_mean'].idxmin()
   #find knn and Mondrian bins for the lowest PI_mean per config
   all_cps = df_cfg_results_all.loc[min_PI_mean_idx]
   all_cps = all_cps.reset_index(drop=True)

   # Call the plot function with the specified results path
   cps_result_plots(all_cps, args.pred_type_units, args.gen_all_plts, results_path, test_data, feat_cols, args.rmse_quant_flag)   
   '''
   
