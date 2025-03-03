
import pandas as pd
import matplotlib.pyplot as plt
import argparse


def plot_config_pi_means(df_cfg_results_all, results_path=None, title=None, metric=None,units=None):
    df_cfg_results_all.reset_index(drop=True,inplace=True)
    min_PI_mean_idx = df_cfg_results_all.groupby('config')['PI_mean'].idxmin()
    #find knn and bins for the lowest PI_mean per config
    mins_knn_bins = df_cfg_results_all.loc[min_PI_mean_idx, ['config','PI_mean','knn', 'bins']]
    #filter rows to only contain those configs
    filtered_df = pd.merge(df_cfg_results_all, mins_knn_bins, on=['config', 'knn', 'bins'])
    
    
    # Group by 'config' and aggregate
    grouped_means = filtered_df.groupby('config').agg({
        'eff_coverage': 'mean',  # Apply the custom function for eff_coverage
        'PI_mean_x': ['mean', 'std', 'count'],  # Calculate mean and std deviation of PI_mean
        'PI_std': 'mean',             # Calculate mean of PI_std
        'knn': 'min',
        'bins': 'min'
    })
    
    grouped_means.columns = ['_'.join(col).strip() for col in grouped_means.columns.values]
    #add index columns to make config its own column
    grouped_means.reset_index(inplace=True)
    #reverse order
    grouped_means = grouped_means.iloc[::-1]#.reset_index(drop=True)
    grouped_means = grouped_means.rename(columns={'PI_mean_x_mean': 'PI_mean_mean', 'PI_mean_x_std':'PI_mean_std'})
    grouped_means['eff_coverage_mean'] = round(grouped_means['eff_coverage_mean'],1).values

    #handle the std = NaN if count is 1    
    # Identify where count is 1
    mask = grouped_means['PI_mean_x_count'] == 1
    # Replace NaN with 0 in 'PI_mean_std' where 'PI_mean_count' is 1
    grouped_means.loc[mask, 'PI_mean_std'] = grouped_means.loc[mask, 'PI_mean_std'].fillna(0)
    
    grouped_means['PI_mean_std'] = round(grouped_means['PI_mean_std'],1).values
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)  # Create a figure and an axes.
    ax.grid(True)
    ymax =max(grouped_means['PI_mean_mean']+grouped_means['PI_mean_std'] + 0.8)
    # Plotting the bar graph with error bars
    # Modified code with conditional inclusion of bins_min
    bars = ax.bar(
        grouped_means['config'] + "\n knn: " + grouped_means['knn_min'].astype(str) +
        grouped_means['config'].apply(lambda x: ' bins: '+str(grouped_means.loc[x == grouped_means['config'], 'bins_min'].values[0]) if 'mond' in x else '') +
        "\ncoverage: " + grouped_means['eff_coverage_mean'].astype(str) + '%\nstd: '+grouped_means['PI_mean_std'].astype(str)+  ' over N='+grouped_means['PI_mean_x_count'].astype(str), 
        grouped_means['PI_mean_mean'], 
        yerr=grouped_means['PI_mean_std'],  # Error bars
        capsize=4,  # Caps on the error bars
        color='lightblue',  # Color of the bars
        alpha=1.0,
        error_kw={'elinewidth': 2, 'ecolor': 'black'}  # Error bar widths and color
    )
    # Adding the value labels on top and to the left of the error bars
    for bar, mean, std in zip(bars, grouped_means['PI_mean_mean'], grouped_means['PI_mean_std']):
        label_x_position = bar.get_x() + bar.get_width() / 2 - 0.1  # Adjust X position here
        label_y_position = bar.get_height()+ 0.1  # Adjust Y position slightly above the error bar
        ax.text(label_x_position, label_y_position, f"{mean:.1f}", ha='right', va='bottom', fontsize=10, color='black')
    
    # Adding labels and title
    ax.set_xlabel('Configuration and Effective Coverage', fontsize=12)
    ax.set_ylabel('Mean '+metric+' Width ['+units+']', fontsize=12)
    ax.set_title("Mean Widths of "+metric+" Prediction Intervals (PI) by Configuration\n"+title)  
    plt.xticks(rotation=45)  # Rotate x labels for better readability
    plt.tight_layout()  # Adjust layout to make room for the rotated x-labels
    
    if results_path is not None:
        filename = title.replace(' ','_')
        plt.savefig(results_path+'/'+filename+'.png', dpi=300, bbox_inches='tight')
      
    # Show the plot
    #plt.show()
    plt.clf()
    return   


def plot_results(results_path, metric, units):
    
    df_cfg_results_all = pd.read_pickle(results_path+'/all_cfg_results.pkl')
    df_cfg_results_all = df_cfg_results_all.rename(columns={'act_conf':'eff_coverage'})
    plot_config_pi_means(df_cfg_results_all, results_path, title='all configs',metric=metric,units=units)
    
    #redo with only trial with > 89% effective coverage
    df_cfg_results_all = df_cfg_results_all[df_cfg_results_all.eff_coverage > 89]
    
    if df_cfg_results_all.empty:
        print("No trial with more than 89% effective coverage")
    else:
        plot_config_pi_means(df_cfg_results_all, results_path, title='means of all configs with effective converage grtr 89',metric=metric,units=units)

#guard to prevent execution during import
if __name__ == "__main__":
   parser = argparse.ArgumentParser(description="Plot results based on path.")
   parser.add_argument("results_path", type=str, help="Path to the directory containing results pkl.")
   
   # Parse command-line arguments
   args = parser.parse_args()

   # Call the plot function with the specified results path
   plot_results(args.results_path)
