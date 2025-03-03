import ipdb
import time
import sys, os, argparse
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Get the directory path for the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Append the parent directory to the system path
sys.path.append(parent_dir)
runtime_env = {"env_vars": {"PYTHONPATH": parent_dir}}

from common.cps import cps_regression
from common import plotting

import warnings
import ray
ray.shutdown()
ray.init()


def load_rsrp_data(root_data_dir,select_model, select_features):

    if (select_model == 'full'):
        df_train = pd.read_parquet(root_data_dir+'cd467019_dtu_rsrp_sat_2020_training_preds.parquet', engine='pyarrow')
        df_cal =pd.read_parquet(root_data_dir+'cd467019_dtu_rsrp_sat_2020_cp_cal_preds.parquet', engine='pyarrow')
        df_test = pd.read_parquet(root_data_dir+'cd467019_dtu_rsrp_sat_2020_eval_preds.parquet', engine='pyarrow')
    
    elif (select_model == 'image_only'):  
        df_train = pd.read_parquet(root_data_dir+'17a0af53_dtu_rsrp_sat_2020_training_preds.parquet', engine='pyarrow')
        df_cal =pd.read_parquet(root_data_dir+'17a0af53_dtu_rsrp_sat_2020_cp_cal_preds.parquet', engine='pyarrow')
        df_test = pd.read_parquet(root_data_dir+'17a0af53_dtu_rsrp_sat_2020_eval_preds.parquet', engine='pyarrow')
    else:
        raise Exception("Bad select_model choice: "+str(select_model))
        
    
    col_names = df_train.columns.to_list()
    if (select_features == 'ext_num'):
        feat_cols = ['Longitude','Latitude','Distance','Distance_x','Distance_y','PCI_64','PCI_65','PCI_302','pathloss']
    elif (select_features.startswith('int')):    
        tab_cols = [x for x in col_names if 'tab' in x]
        img_cols = [x for x in col_names if 'img' in x]
        if select_features == 'int_num_image':  
            feat_cols = tab_cols+ img_cols
        elif select_features == 'int_num':
            feat_cols = tab_cols
        elif select_features =='int_image':      
            feat_cols = img_cols
        else:
            raise Exception("Bad select_features choice: "+str(select_features))
    else:
        raise Exception("Bad select_features choice: "+str(select_features))
        
        
    train_data_ref = ray.put(df_train)
    cal_data_ref = ray.put(df_cal)
    test_data_ref = ray.put(df_test)
    # Creating a list of ObjectRefs by invoking process_data
    data_refs =[train_data_ref,cal_data_ref, test_data_ref]    
        
    return data_refs, feat_cols


def proc_data(seed, data_refs, feat_cols):
    train_data,cal_data, test_data = ray.get(data_refs)
    
    #columns
    target = 'targets'
    prediction = 'pred'
    
    #get relevant data splits
    df_train = pd.DataFrame(columns=feat_cols)
    df_train[feat_cols] = train_data[feat_cols]
    df_train['targets'] = train_data[target]
    df_train['pred'] = train_data[prediction]

    df_cal = pd.DataFrame(columns=feat_cols)
    df_cal[feat_cols] = cal_data[feat_cols]
    df_cal['targets'] = cal_data[target]
    df_cal['pred'] = cal_data[prediction]


    df_test = pd.DataFrame(columns=feat_cols)
    df_test[feat_cols] = test_data[feat_cols]
    df_test['targets'] = test_data[target]
    df_test['pred'] = test_data[prediction]   

    return df_train, df_cal, df_test, feat_cols

@ray.remote(runtime_env=runtime_env)
def conformal_prediction(data_refs,seed,knn_num,bin_num, targ_strangeness, feat_cols, req_conf_intervals, include_pred_metadata=False):  
    #set the seed ensures reproducibility when using CREPES
    np.random.seed(seed)
    
    #mute warnings caused by the approx 10% that are not in the effective coverage
    warnings.filterwarnings("ignore", category=UserWarning, module="mapie")
    #mute warning about MinMaxScalar and feature names
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

    df_train, df_cal, df_test,feat_cols = proc_data(seed, data_refs, feat_cols)
    #start the multimodal conformal regression
    df_results = cps_regression(seed, df_train, df_cal, df_test,feat_cols, knn_num,bin_num,targ_strangeness, req_conf_intervals, include_pred_metadata)

    return df_results
###########
#main loop
def proc_and_plt_conformal_predictions(configs,select_model,seed_list):
    root_data_dir = '../../data/rsrp/'
    
    for model in select_model:
      for config in configs:
        if model == 'image_only' and config != 'int_image':
                continue  #'image_only' model only has 'int_image' config
        print(f"Model: {model}, Config: {config}")
        
        start_time = time.time()
        data_refs, feat_cols = load_rsrp_data(root_data_dir, model, config)
        
        bin_nums = range(10, 100, 10)
        knn_nums = range(10, 100, 10)  
    
        req_conf_intervals = [95,90]
        custom_difficulty_est = [True, False]
        ray_results = [conformal_prediction.remote(data_refs, seed, knn_num, bin_num, targ_strangeness, \
                                                   feat_cols, req_conf_intervals=req_conf_intervals, include_pred_metadata=True)
                       for seed in seed_list
                       for knn_num in knn_nums
                       for bin_num in bin_nums
                       for targ_strangeness in custom_difficulty_est]
        
        list_of_dfs = ray.get(ray_results)
        df_cfg_results_all = pd.concat(list_of_dfs)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print("Execution time:", round(execution_time/60,2), "minutes")

        # Create unique output directory name
        now = datetime.now()
        dir_name = now.strftime("%m-%d_%Hh%Mm%Ss")
        # Specify your base directory
        base_path = '../../results/rsrp/'+model+'_'+config
        results_path = os.path.join(base_path, dir_name)
        # Make the directory if it does not exist
        os.makedirs(results_path, exist_ok=True)
    
        print(f"Results dir created at: {results_path}")
        #save experiments:
        df_cfg_results_all.to_pickle(results_path+'/all_cfg_results.pkl',protocol=4)
        #plot aggregated experiments
        plotting.plot_results(results_path,metric='RSRP',units='dB')


if __name__ == "__main__":
  '''
  example how to run
  /home/ray/cp_mm_targ_strg/src$ python main_rsrp.py --seed_list 56422 536073 9164 9847117 92631321
  '''
    
  parser = argparse.ArgumentParser(description="RSRP prediction intervals")
  parser.add_argument("--seed_list", type=int, nargs='+', help="List of seeds for the experiments")

  # Parse command-line arguments
  args = parser.parse_args()
  
  configs = ['ext_num','int_num_image','int_image','int_num'] 
  select_model = ['image_only','full']

  proc_and_plt_conformal_predictions(configs,select_model,args.seed_list)


