import ipdb
import time
import sys, os, argparse
import pandas as pd
import numpy as np
import json
from datetime import datetime

# Get the directory path for the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Append the parent directory to the system path
sys.path.append(parent_dir)
runtime_env = {"env_vars": {"PYTHONPATH": parent_dir}}

from common.cps import cps_regression
from common import plotting


import ray
ray.shutdown()

#double check this is fine for map-based
num_total_cpus =  4 
ray.init(num_cpus=num_total_cpus)

def load_data_mlpl(root_data_dir):
    train_data = pd.read_csv(root_data_dir+'CNN_train_uncertainty_data_.csv')
    cal_data = pd.read_csv(root_data_dir+'CNN_cal_uncertainty_data_.csv')
    test_data = pd.read_csv(root_data_dir+'CNN_test_uncertainty_data_.csv')
    
    #columns
    #feat_cols = ['actual_path_loss', 'predicted_path_loss']

    train_data = train_data.rename(columns={'actual_path_loss': 'targets', 'predicted_path_loss': 'pred'})
    cal_data = cal_data.rename(columns={'actual_path_loss': 'targets', 'predicted_path_loss': 'pred'})
    test_data = test_data.rename(columns={'actual_path_loss': 'targets', 'predicted_path_loss': 'pred'})
   
    col_names = train_data.columns.to_list()
    feat_cols = [x for x in col_names if 'internal_feature' in x]

    #ipdb.set_trace()
    train_data_ref = ray.put(train_data)
    cal_data_ref = ray.put(cal_data)
    test_data_ref = ray.put(test_data)
    # Creating a list of ObjectRefs by invoking process_data
    data_refs =[train_data_ref,cal_data_ref,test_data_ref]
    
    return data_refs, feat_cols

def process_and_sample(data, feat_cols, target_col, pred_col, sample_frac, seed):
    # Create DataFrame and sample rows efficiently
    df = data[feat_cols + [target_col, pred_col]].copy()  # Avoid unnecessary column assignments
    df = df.sample(frac=sample_frac, random_state=seed).reset_index(drop=True)
    return df

def proc_data(seed, data_refs, feat_cols):
    # Fetch the data from Ray references
    train_data, cal_data, test_data = ray.get(data_refs)
    
    # Column names
    target = 'targets'
    prediction = 'pred'
    sample_frac = 0.5  # As per your original example
    
    # Use helper function to process train, calibration, and test data
    df_train = process_and_sample(train_data, feat_cols, target, prediction, sample_frac, seed)
    #breakpoint() 
    df_cal = process_and_sample(cal_data, feat_cols, target, prediction, 1.0 , seed)
    #breakpoint() 
    df_test = process_and_sample(test_data, feat_cols, target, prediction, sample_frac, seed)
    
    return df_train, df_cal, df_test, feat_cols

@ray.remote(runtime_env=runtime_env)
def conformal_prediction(data_refs,seed,knn_num,bin_num, targ_strangeness, feat_cols, req_conf_intervals, include_pred_metadata=False):  
    np.random.seed(seed)

    df_train, df_cal, df_test,feat_cols = proc_data(seed, data_refs, feat_cols)
    #start the multimodal conformal regression
    df_results = cps_regression(seed, df_train, df_cal, df_test,feat_cols, knn_num,bin_num,targ_strangeness, req_conf_intervals, include_pred_metadata)

    return df_results

def proc_and_plt_conformal_predictions(search_params):
    root_data_dir = '../../data/map_based_mlpl/'
    
    for config in search_params['configs']:  # Use square brackets        
        start_time = time.time()
        
        data_refs, feat_cols = load_data_mlpl(root_data_dir) 
        
        ray_results = [conformal_prediction.remote(data_refs, seed, knn_num, bin_num, targ_strangeness, \
                                                   feat_cols, req_conf_intervals=search_params['req_conf_intervals'],\
                                                    include_pred_metadata = search_params['include_pred_metadata'])
                       for seed in search_params['seed_list']
                       for knn_num in search_params['knn_nums']
                       for bin_num in search_params['bin_nums']
                       for targ_strangeness in search_params['targ_strg']]
        
        list_of_dfs = ray.get(ray_results)
        df_cfg_results_all = pd.concat(list_of_dfs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        print("Execution time:", round(execution_time/60,2), "minutes")

        # Create unique output directory name
        now = datetime.now()
        dir_name = now.strftime("%m-%d_%Hh%Mm%Ss")
        # Specify your base directory
        base_path = '../../results/map_based_mlpl/'+config
        results_path = os.path.join(base_path, dir_name)
        # Make the directory if it does not exist
        os.makedirs(results_path, exist_ok=True)
        
        print(f"Results dir created at: {results_path}")
        #save experiments:
        df_cfg_results_all.to_pickle(results_path+'/all_cfg_results.pkl',protocol=4)
        
        if search_params['specific_config']:
            #ipdb.set_trace()
            json.dump(feat_cols, open(results_path+'/feat_columns.csv', 'w'), ensure_ascii=False)
        
        plotting.plot_results(results_path,metric='Path Loss', units='dB')
        
def create_parser():
    parser = argparse.ArgumentParser(description="Experiment Configuration Parser")
    
    # Create a mutually exclusive group for seed specification
    seed_group = parser.add_mutually_exclusive_group(required=True)
    
    # Option 1: Seed list
    seed_group.add_argument("--seed_list", type=int, nargs='+', 
                            help="List of seeds for the experiments")
    
    # Option 2: Specific configuration flag
    seed_group.add_argument('--specific_config', action='store_true', 
                            help='Flag to indicate specific configuration mode')
    
    # Arguments for specific configuration
    parser.add_argument("--seed", type=int, 
                        help="Single seed (required with --specific_config)")
    parser.add_argument("--bin_num", type=int, 
                        help="Bin number (required with --specific_config)")
    parser.add_argument("--knn_num", type=int, 
                        help="KNN number (required with --specific_config)")
    parser.add_argument("--req_conf_interval", type=int, 
                        help="Requested confidence interval (required with --specific_config)")
    parser.add_argument("--targ_strg", type=str, choices=['True', 'False'], 
                    help="Target Strangeness difficulty estimator (required with --specific_config)")
    parser.add_argument("--config", type=str, 
                        help="Configuration to use (required with --specific_config)")
    
    return parser

def validate_specific_config(args):
    """
    Validate that all specific configuration parameters are provided 
    when --specific_config is used
    """
    if args.specific_config:
        required_args = ['seed', 'bin_num', 'knn_num', 
                         'req_conf_interval']
        
        # Check if any of the required arguments are missing
        missing_args = [arg for arg in required_args if getattr(args, arg) is None]
        
        if missing_args:
            raise ValueError(
                f"The following arguments are required with --specific_config: {', '.join(missing_args)}"
            )
    
    return args
        

if __name__ == "__main__":
    # Create parser
    parser = create_parser()
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate specific configuration
    try:
        args = validate_specific_config(args)
    except ValueError as e:
        parser.error(str(e))
    
    args.targ_strg = args.targ_strg == 'True'
    
    if args.specific_config:
        # Specific configuration
        search_structure = {
            'seed_list': [args.seed],
            'bin_nums': [args.bin_num],
            'knn_nums': [args.knn_num],
            'req_conf_intervals': [args.req_conf_interval],
            'targ_strg': [True, False],
            'configs': ['int_cnn'],
            'include_pred_metadata': True,
             'specific_config': args.specific_config
        }
        
    else:
        print('Running comprehensive hyperparametre search')
        # Default behavior with seed list
        search_structure = {
            'seed_list': args.seed_list,
            'bin_nums': range(10, 100, 20),
            'knn_nums': range(10, 100, 20),
            'req_conf_intervals': [95],
            'targ_strg': [True, False],
            'configs': ['int_cnn'],
            'include_pred_metadata': False,
            'specific_config': args.specific_config
        }
        
    proc_and_plt_conformal_predictions(search_structure)



