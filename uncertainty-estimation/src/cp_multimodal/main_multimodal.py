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
import multiprocessing
import ray
ray.shutdown()

num_cpus = max(1, multiprocessing.cpu_count() - 1)
ray.init(num_cpus=num_cpus)

def load_mm_data(feat_types,root_data_dir):
    df_train = pd.read_csv(root_data_dir+'train_tab_trans_feats.csv')
    df_val = pd.read_csv(root_data_dir+'val_tab_trans_feats.csv')
    df_test =pd.read_csv(root_data_dir+'test_tab_trans_feats.csv')

    
    df_train = df_train.rename(columns={'labels': 'targets', 'preds': 'pred'})
    df_val = df_val.rename(columns={'labels': 'targets','preds': 'pred'})
    df_test = df_test.rename(columns={'labels': 'targets','preds': 'pred'})
   
    col_names = df_train.columns.to_list()
    bert_cols = [x for x in col_names if 'bert' in x]
    num_cols = [x for x in col_names if 'num' in x]
    cat_cols = [x for x in col_names if 'cat' in x]
    
    if (feat_types == 'bert_cat_num'):
        feat_cols = bert_cols+cat_cols+num_cols
    elif (feat_types == 'bert'):    
        feat_cols = bert_cols
    elif (feat_types == 'cat'):    
        feat_cols = cat_cols
    elif (feat_types == 'num'):    
        feat_cols = num_cols
    else:
        raise ValueError("unsupported feature type"+feat_types)

    train_data_ref = ray.put(df_train)
    cal_test_data_ref = ray.put(df_test)
    # Creating a list of ObjectRefs by invoking process_data
    data_refs =[train_data_ref,cal_test_data_ref]
    return data_refs, feat_cols

def load_mm_external_data(root_data_dir):
    
    df_train = pd.read_csv(root_data_dir+'train.csv')
    df_train['pred'] = pd.read_csv(root_data_dir+'train_tab_trans_feats.csv')['preds']
    df_val = pd.read_csv(root_data_dir+'val.csv')
    df_val['pred'] = pd.read_csv(root_data_dir+'val_tab_trans_feats.csv')['preds']
    df_test =pd.read_csv(root_data_dir+'test.csv')
    df_test['pred'] =pd.read_csv(root_data_dir+'test_tab_trans_feats.csv')['preds']
    
    df_train = df_train.rename(columns={'price': 'targets', 'preds': 'pred'})
    df_val = df_val.rename(columns={'price': 'targets','preds': 'pred'})
    df_test = df_test.rename(columns={'price': 'targets','preds': 'pred'})
    
    col_data = [] # your list with json objects (dicts)

    with open(root_data_dir+'column_info.json') as json_file:
        col_data = json.load(json_file)
    
    num_cols = col_data['num_cols']
    #cannot using non numeric columns
    #cat_cols = col_data['cat_cols']
    #text_cols = col_data['text_cols']
    
    feat_cols = num_cols
    
    df_train = df_train[feat_cols+['targets','pred']].dropna()
    df_val = df_val[feat_cols+['targets','pred']].dropna()
    df_test = df_test[feat_cols+['targets','pred']].dropna()
    
    df_cal_test = df_test
    train_data_ref = ray.put(df_train)
    cal_test_data_ref = ray.put(df_cal_test)
    # Creating a list of ObjectRefs by invoking process_data
    data_refs =[train_data_ref,cal_test_data_ref]

    return data_refs, feat_cols


def proc_data(seed, data_refs, feat_cols):
    train_data,cal_test_data = ray.get(data_refs)
    
    #Randomly sample 1000 points
    #random_state ensures reproducibility
    cal_data = cal_test_data.sample(n=1000, random_state=seed)  

    # Create the test DataFrame from the remaining rows
    test_data = cal_test_data.drop(cal_data.index)  # Drop the sampled rows by index
    
    #columns
    target = 'targets'
    prediction = 'pred'
    
    #get relevant data splits
    df_train = pd.DataFrame(columns=feat_cols)
    df_train[feat_cols] = train_data[feat_cols]
    df_train['targets'] = train_data[target].values
    df_train['pred'] = train_data[prediction].values


    df_cal = pd.DataFrame(columns=feat_cols)
    df_cal[feat_cols] = cal_data[feat_cols]
    df_cal['targets'] = cal_data[target].values
    df_cal['pred'] = cal_data[prediction].values


    df_test = pd.DataFrame(columns=feat_cols)
    df_test[feat_cols] = test_data[feat_cols]
    df_test['targets'] = test_data[target].values
    df_test['pred'] = test_data[prediction].values     

    return df_train, df_cal, df_test,feat_cols

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

def proc_and_plt_conformal_predictions(search_params):
    root_data_dir = '../../data/multimodal/'
 
    for config in search_params['configs']:  # Use square brackets
        feat_src, feat_types = config.split('_', 1)
        
        start_time = time.time()
        
        if (feat_src == 'int'):
            data_refs, feat_cols = load_mm_data(feat_types, root_data_dir)
        elif (feat_src == 'ext'): 
            data_refs, feat_cols = load_mm_external_data(root_data_dir)
        else:
            raise ValueError("unsupported feature source")
        #ipdb.set_trace()
        ray_results = [conformal_prediction.remote(data_refs, seed, knn_num, bin_num, targ_strangeness, \
                                                   feat_cols, search_params['req_conf_intervals'], 
                                                   include_pred_metadata=search_params['include_pred_metadata'])
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
        base_path = '../../results/multimodal/'+config
        results_path = os.path.join(base_path, dir_name)
        # Make the directory if it does not exist
        os.makedirs(results_path, exist_ok=True)
        #save experiments:
        print(f"Results dir created at: {results_path}")
        df_cfg_results_all.to_pickle(results_path+'/all_cfg_results.pkl',protocol=4)
       
        if search_params['specific_config']:
            #ipdb.set_trace()
            #json.dump(feat_cols, open(results_path+'/feat_columns.csv', 'w'))
            json.dump(feat_cols, open(results_path+'/feat_columns.csv', 'w'), ensure_ascii=False)
            #feat_cols.to_csv('feat_columns.csv')
        #plot aggregated experiments  
        plotting.plot_results(results_path,metric='Price',units='$')

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
                         'req_conf_interval', 'targ_strg', 'config']
        
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
        ipdb.set_trace()
    
    args.targ_strg = args.targ_strg == 'True'
    # Your main logic here
    if args.specific_config:
        # Specific configuration
        search_structure = {
            'seed_list': [args.seed],
            'bin_nums': [args.bin_num],
            'knn_nums': [args.knn_num],
            'req_conf_intervals': [args.req_conf_interval],
            'targ_strg': [args.targ_strg],
            'configs': [args.config],
            'include_pred_metadata': True,
             'specific_config': args.specific_config
        }
        
    else:
        print('Running comprehensive hyperparametre search')
        # Default behavior with seed list
        search_structure = {
            'seed_list': args.seed_list,
            'bin_nums': range(10, 60, 10),
            'knn_nums': range(10, 100, 10),
            'req_conf_intervals': [90],
            'targ_strg': [True, False],
            'configs': ['ext_num','int_bert_cat_num','int_bert','int_cat','int_num'],
            'include_pred_metadata': False,
            'specific_config': args.specific_config
        }

    proc_and_plt_conformal_predictions(search_structure)
