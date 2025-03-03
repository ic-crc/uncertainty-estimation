## Description (English)
What is this project?

This Uncertainty Estimation project creates prediction intervals to estimate uncertainty in multimodal and communications problems by leveraging internal neural network features from complex architectures, particularly when dealing with unstructured data like images and text.

How does it work?

The methodology leverages internal features from neural network architectures, such as CNNs and transformers, at points where multimodal information converges. These internal features are then incorporated with conformal prediction models through the Conformal Classifiers Regressors and Predictive Systems (CREPES) framework to construct prediction intervals. The model is first trained on an initial dataset, after which the conformal predictors are calibrated using a calibration set, enabling robust uncertainty estimation. Additionally, a novel difficulty estimator for conformal prediction aims to enhance prediction intervals by evaluating how atypical a prediction is within the context of its nearest neighbors' target distribution.

Who will use this project?

Researchers and practitioners in machine learning and data science who need to estimate uncertainty in predictive models, particularly those handling complex multimodal unstructured data such as images and unstructured text.

What is the goal of this project?

The goal is to extend the applicability of conformal prediction techniques to multimodal regression tasks, providing robust uncertainty estimation for models utilizing diverse data inputs.

## Description (Français)

Quel est ce projet ?

Ce projet de estimation de l'incertitude a pour objectif de créer des intervalles de prédiction afin de mesurer l'incertitude dans des contextes multimodaux et de communication. Il s'appuie sur les caractéristiques internes des réseaux neuronaux issus d'architectures complexes, notamment lorsqu'il s'agit de données non structurées, telles que les images et le texte.

Comment cela fonctionne-t-il ?

La méthodologie utilise les caractéristiques internes des architectures de réseaux neuronaux, tels que les CNN et les transformeurs, au niveau des points de convergence des informations multimodales. Ces caractéristiques internes sont ensuite intégrées aux modèles de prédiction conforme via le cadre des Conformal Classifiers Regressors and Predictive Systems (CREPES) pour construire des intervalles de prédiction. Le modèle est d'abord entraîné sur un ensemble de données initial, après quoi les prédicteurs conformes sont calibrés à l'aide d'un ensemble de calibration, permettant une estimation robuste de l'incertitude. De plus, un nouvel estimateur de difficulté pour la prédiction conforme vise à améliorer les intervalles de prédiction en évaluant à quel point une prédiction est atypique dans le contexte de la distribution cible de ses voisins les plus proches.

Qui utilisera ce projet?

Les chercheurs et praticiens en apprentissage automatique et en science des données qui ont besoin de estimer l'incertitude dans les modèles prédictifs, notamment ceux traitant des données non structurées complexes multimodales telles que des images et du texte non structuré.

Quel est l’objectif de ce projet?

L'objectif est d'étendre l'applicabilité des techniques de prédiction conforme aux tâches de régression multimodale, en fournissant une estimation robuste de l'incertitude pour les modèles utilisant des entrées de données diversifiées.

## What is Conformal Prediction?
For an introduction to conformal prediction, please refer to: <br>
https://medium.com/low-code-for-advanced-data-science/conformal-prediction-theory-explained-14a35226df80 . 

More information on the CREPES framework can be found in this paper:<br> 
https://proceedings.mlr.press/v179/bostrom22a/bostrom22a.pdf <br>
and this jupyter notebook:<br>
https://github.com/henrikbostrom/crepes/blob/main/docs/crepes_nb.ipynb .
 
## Uncertainty Estimation

The experiments were performed on an AWS  m6i.4x (16 cores and 64GB RAM) using 22.04.1 Ubuntu running Docker version 24.0.5.

Build the docker:
```
	docker build -t uncertainty-estimation:latest .
```

Make results directory:
```
	cd uncertainty-estimation

	mkdir results
    
        cd ..
```

Note: this needs to be owned by the same user as cloned repository directory and is used to mount to the docker to get the results back to the host to view, otherwise you will get a permission error.

Run the docker:
```
	docker run -it --rm \
	-v $(pwd)/uncertainty-estimation:/home/ray/uncertainty-estimation \
	uncertainty-estimation:latest /bin/bash
```
# **Target Strangeness: A Novel Conformal Prediction Difficulty Estimator**
This paper introduces 'Target Strangeness,' a method for estimating prediction difficulty by evaluating how atypical a prediction is relative to its nearest neighbours' target distribution.

**For the target strangeness results from arxiv.org/abs/2410.19077:**

Run the baseline
```
	cd /home/ray/uncertainty-estimation/src/targ_strg/

	/home/ray/uncertainty_quant/targ_strg$ python crepes_nb_targ_strg_baseline.py
```
To reproduce the error bar plots results in the paper within the docker shell:
```
        cd /home/ray/uncertainty-estimation/data/mlpl
	cat mlpl_train* > mlpl_train.parquet

	cd /home/ray/uncertainty-estimation/src/targ_strg
	
	#MLPL error plots (~20 min):
	/home/ray/uncertainty-estimation/src/targ_strg$ python main_mlpl.py --seed_list 56422 536073 9164 9847117 92631321
```

# *Evaluating Conformal Prediction with Optimal Memory Usage*

The following 2 sections: **Conformal Prediction for Multimodal Regression** and **Uncertainty Estimation for Path Loss and Radio Metric Models** required a two-step process to evaluate **conformal prediction intervals** in order to reduce memory usage. First, we run a broad hyperparameter search to find the configuration that minimizes interval width while maintaining coverage above 89%, just storing the essential metrics. Then, looking at the results we can determine the optimal configuration. With this information, we re-run it with `--specific_config` to store the necessary data needed to compute and plot the prediction intervals. This allows for the process to be completed without requiring large amounts of memory.

# **Conformal Prediction for Multimodal Regression**
This paper presents a novel methodology to extend conformal prediction to multimodal regression by utilizing internal features from neural network architectures that process unstructured data such as images and text.

**For the Conformal Prediction for Multimodal Regression results from https://arxiv.org/pdf/2410.19653:**
	
To reproduce the error bar plots results in the paper within the docker shell:
```	
	cd /home/ray/uncertainty-estimation/src/cp_multimodal

	#RSRP error plots (~1h40min):
	/home/ray/uncertainty-estimation/src/cp_multimodal$ python main_rsrp.py --seed_list 56422 536073 9164 9847117 92631321

	#Multimodal Airbnb errors plot (~2.5h):
        cd /home/ray/uncertainty-estimation/data/multimodal 
        /home/ray/uncertainty-estimation/data/multimodal$ unzip -j mm_data.zip -d .
        
        cd /home/ray/uncertainty-estimation/src/cp_multimodal 
	/home/ray/uncertainty-estimation/src/cp_multimodal$ python main_multimodal.py --seed_list 56422 536073 9164 9847117 92631321

	cd /home/ray/uncertainty-estimation/results/mutimodal$ inspect the all_cfg_results.pkl results in a dataframe, and select the desired model, and record the hyperparameters
	go back to the src directory and rerun main_multimodal.py with the --specific_config flag to include the prediction metadata, required to run predictions.

	For example:
	#ext_num
	/home/ray/uncertainty-estimation/src/cp_multimodal$ python main_multimodal.py --specific_config --config ext_num --seed 9164 --bin_num 20 --knn_num 20 --req_conf_interval 90 --targ_strg False

	#int_bert
	/home/ray/uncertainty-estimation/src/cp_multimodal$ python main_multimodal.py --specific_config --config int_bert --seed 9164 --bin_num 10 --knn_num 10 --req_conf_interval 90 --targ_strg False 

	#int_bert_cat_num
	/home/ray/uncertainty-estimation/src/cp_multimodal$ python main_multimodal.py --specific_config --config int_bert_cat_num --seed 9164 --bin_num 10 --knn_num 70 --req_conf_interval 90 --targ_strg True 

	#int_cat
	/home/ray/uncertainty-estimation/src/cp_multimodal$ python main_multimodal.py --specific_config --config int_cat --seed 9164 --bin_num 10 --knn_num 30 --req_conf_interval 90 --targ_strg True 

	#int_num
	/home/ray/uncertainty-estimation/src/cp_multimodal$ python main_multimodal.py --specific_config --config int_num --seed 9164 --bin_num 10 --knn_num 20 --req_conf_interval 90 --targ_strg True 
```
The following are examples on how to reproduce the prediction interval example plots. Your ```--results_file_path all_cfg_results.pkl``` **will be different** than the ones shown here. Navigate to the appropriate results folder and find the most recent .pkl file and copy the name of the folder into the below commands.

For example: python ../src/common/cps_result_plots.py --results_file_path ./multimodal/int_bert/**02-18_07h08m26s**/all_cfg_results.pkl --test_data_dir ../data/multimodal/ --test_data_fname test_tab_trans_feats.csv --pred_type_units "Price [$]"

You will also need to adjust the command for each different configuration, in order to produce the plots for each


```
	cd /home/ray/uncertainty-estimation/results

	#multimodal int_bert example
	uncertainty-estimation/results$ python ../src/common/cps_result_plots.py --results_file_path ./multimodal/int_bert/02-18_07h08m26s/all_cfg_results.pkl --test_data_dir ../data/multimodal/ --test_data_fname test_tab_trans_feats.csv --pred_type_units "Price [$]"
	
```

Go to the ```./results``` host directory of interest to view the prediction interval plots.


# **Uncertainty Estimation for Path Loss and Radio Metric Models**
This paper utilizes Conformal Prediction Systems (CPS) to create reliable uncertainty estimates for machine learning-based radio metric models and 2-D map-based path loss models.

**For the Uncertainty Estimation for Path Loss and Radio Metric Models from arxiv.org/abs/2501.06308:**

Only the map_based_mlpl code and data is available as it is based on the open source ITU-R U.K. OFCom drive test dataset
"U.K. mobile measurment data for frequencies below 6 GHz", Dec. 2023: https://www.itu.int/md/R15-WP3K-C-0294.


To reproduce the error bar plots results in the paper within the docker shell:
```	
	Download all files in https://doi.org/10.5281/zenodo.14926307 to here /home/ray/uncertainty-estimation/data/map_based_mlpl

	cd /home/ray/uncertainty-estimation/src/cp_pl_radio_metrics

	#map-based MLPL (Machine Learning Path Loss) error plots (~2 hrs):
	/home/ray/uncertainty_estimation/src/cp_pl_radio_metric$ python main_map_based_mlpl.py --seed_list 56422 536073 9164 9847117 92631321

	#Run specific config to get the plot shown in the paper (~5 min):
	/home/ray/uncertainty_estimation/src/cp_pl_radio_metric$ python main_map_based_mlpl.py --specific_config --config int_cnn --seed 92631321 --bin_num 10 --knn_num 70 --req_conf_interval 95

	#map_based_mlpl_analysis
        cd /home/ray/uncertainty_estimation/src/cp_pl_radio_metric/map_based_analysis
	/home/ray/uncertainty_estimation/src/cp_pl_radio_metric/map_based_analysis$ python map_based_mlpl_analysis.py


	#The radio metrics data is closed-source.
```
On the host go to ```./results``` directory to view the error bar plots.

The following are examples on how to reproduce the prediction interval example plots. Your ```--results_file_path all_cfg_results.pkl``` **will be different** than the ones shown here. Navigate to the appropriate results folder and find the most recent .pkl file and copy the name of the folder into the below commands.

For example: python ../src/common/cps_result_plots.py --results_file_path ./map_based_mlpl/int_cnn/**02-19_12h46m59s**/all_cfg_results.pkl --test_data_dir ../data/map_based_mlpl/ --test_data_fname CNN_test_uncertainty_data_.csv --pred_type_units "Path Loss [dB]" --rmse_quant_flag
```
	cd /home/ray/uncertainty-estimation/results

	#map based MLPL
	uncertainty-estimation/results$ python ../src/common/cps_result_plots.py --results_file_path ./map_based_mlpl/int_cnn/02-19_12h46m59s/all_cfg_results.pkl --test_data_dir ../data/map_based_mlpl/ --test_data_fname CNN_test_uncertainty_data_.csv --pred_type_units "Path Loss [dB]" --rmse_quant_flag

	#The radio metrics data is closed-source.
```

Go to the ```./results``` host directory of interest to view the prediction interval plots.



### If you find anything useful, please cite one or some of the following: ###

**Target Strangeness: A Novel Conformal Prediction Difficulty Estimator**
```
@misc{bose2024targetstrangenessnovelconformal,
      title={Target Strangeness: A Novel Conformal Prediction Difficulty Estimator}, 
      author={Alexis Bose and Jonathan Ethier and Paul Guinand},
      year={2024},
      eprint={2410.19077},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.19077}, 
}
```
**Conformal Prediction for Multimodal Regression**
```
@misc{bose2024conformalpredictionmultimodalregression,
      title={Conformal Prediction for Multimodal Regression}, 
      author={Alexis Bose and Jonathan Ethier and Paul Guinand},
      year={2024},
      eprint={2410.19653},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.19653}, 
}
```
**Uncertainty Estimation for Path Loss and Radio Metric Models**
```
@misc{bose2025uncertaintyestimationpathloss,
      title={Uncertainty Estimation for Path Loss and Radio Metric Models}, 
      author={Alexis Bose and Jonathan Ethier and Ryan G. Dempsey and Yifeng Qiu},
      year={2025},
      eprint={2501.06308},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2501.06308}, 
}

