"""Conformal classifiers, regressors, and predictive systems (crepes) extras

Functions for generating non-conformity scores and Mondrian categories
(bins), and a class for generating difficulty estimates, with and without
out-of-bag predictions.

Author: Henrik Boström (bostromh@kth.se)

Copyright 2024 Henrik Boström

License: BSD 3 clause

"""

#Modified from CREPES 0.5
#https://github.com/henrikbostrom/crepes/blob/fe1f88798e3d1a32e5c9c9ef86aeff51b140ae66/docs/crepes_nb.ipynb

# # More examples

# ## Importing packages
# 
# In the examples below, we will be using the three main classes `ConformalClassifier`, `ConformalRegressor`, and `ConformalPredictiveSystem` from the `crepes` package, as an alternative to using the classes `WrapClassifier` and `WrapRegressor` in the same package, which was illustrated [here](https://crepes.readthedocs.io/en/latest/crepes_nb_wrap.html). In the examples, we will be using a helper class and functions from `crepes.extras` as well as `NumPy`, `pandas`, `matplotlib` and `sklearn`. 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

from crepes import ConformalClassifier, ConformalRegressor, ConformalPredictiveSystem, __version__

knn_targ_strange_flag = 1
from crepes.extras_with_targ_strangeness import hinge, margin, binning, DifficultyEstimator


print(f"crepes v. {__version__}")

np.random.seed(602211023)


# ## Conformal regressors (CR)

# ### Importing and splitting a regression dataset

# Let us import a regression dataset from [www.openml.org](https://www.openml.org) and min-max normalize the targets; the latter is not really necessary, but useful, allowing to directly compare the size of a prediction interval to the whole target range, which becomes 1.0 in this case.



#dataset = fetch_openml(name="house_sales", version=3, parser="auto")

#openml went down, so lets load the data locally:
df = pd.read_csv('../../data/targ_strg/house_sales.csv')

#take out extra column
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

#order the columns to ensure they are the same
feature_names = [
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
    'waterfront', 'view', 'condition', 'grade', 'sqft_above', 
    'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 
    'long', 'sqft_living15', 'sqft_lot15', 'date_year', 'date_month', 'date_day'
]

X = df[feature_names].values.astype(float)

#price is the target
y = df['price'].values.astype(float)

# Create a dataset-like object to mimic OpenML's structure
class DatasetWrapper:
    def __init__(self, data, target, feature_names, target_names, description):
        self.data = pd.DataFrame(data, columns=feature_names)
        self.target = pd.Series(target)
        self.feature_names = feature_names
        self.target_names = target_names
        self.DESCR = description

# Create the dataset
dataset = DatasetWrapper(
    data=X, 
    target=y, 
    feature_names=feature_names,
    target_names=['price'],
    description="House sales dataset from King County"
)
X = dataset.data.values.astype(float)
y = dataset.target.values.astype(float)

#true_max = y.max()
y = np.array([(y[i]-y.min())/(y.max()-y.min()) for i in range(len(y))])
y_model_min = 0
y_model_max = 1#7701000



# We now split the dataset into a training and a test set, and further split the training set into a proper training set and a calibration set. Let us fit a random forest to the proper training set.


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

X_prop_train, X_cal, y_prop_train, y_cal = train_test_split(X_train, y_train,
                                                            test_size=0.25)

learner_prop = RandomForestRegressor(n_jobs=-1, n_estimators=500, oob_score=True) 

learner_prop.fit(X_prop_train, y_prop_train)


### Standard conformal regressors

# Let us create a conformal regressor.


cr_std = ConformalRegressor()


# We may print the object, e.g., to see whether it has been fitted or not.

print(cr_std)


# We will use the residuals from the calibration set to fit the conformal regressor. 

y_hat_cal = learner_prop.predict(X_cal)

residuals_cal = y_cal - y_hat_cal

cr_std.fit(residuals_cal)


# We may now obtain prediction intervals from the point predictions for the test set; 
# here using a confidence level of 99%.


y_hat_test = learner_prop.predict(X_test)

intervals = cr_std.predict(y_hat_test, confidence=0.99)

print(intervals)


# We may request that the intervals are cut to exclude impossible values, in this case below 0 and above 1; below we also use the default 
# confidence level (95%), which further tightens the intervals.


intervals_std = cr_std.predict(y_hat_test, y_min=y_model_min, y_max=y_model_max)

print(intervals_std)


# ### Normalized conformal regressors

# The above intervals are not normalized, i.e., they are all of the same size (at least before they are cut). We could make the intervals more informative through normalization using difficulty estimates; 
# more difficult instances will be assigned wider intervals. We can use a `DifficultyEstimator`, as imported from `crepes.extras`,  for this purpose. It can be used to estimate the difficulty by using k-nearest neighbors in three different ways: i) by the (Euclidean) distances to the nearest neighbors, ii) by the standard deviation of the targets of the nearest neighbors, and iii) by the absolute errors of the k nearest neighbors. 
# 
# A small value (beta) is added to the estimates, which may be given through a (named) argument to the `fit` method; we will just use the default for this, i.e., `beta=0.01`. In order to make the beta value have the same effect across different estimators, we may opt for normalizing the difficulty estimates (using min-max scaling) by setting `scaler=True`. It should be noted that this comes with a computational cost; for estimators based on the k-nearest neighbor, a leave-one-out protocol is employed to find the minimum and maximum distances that are used by the scaler.
# 
# We will first consider just using the first option (distances to the k-nearest neighbors) to produce normalized conformal regressors, using the default number of nearest neighbors, i.e., `k=25`.


de_knn = DifficultyEstimator(knn_target_strangeness=knn_targ_strange_flag)

de_knn.fit(X=X_prop_train, scaler=True)

print(de_knn)

sigmas_cal_knn_dist = de_knn.apply(X_cal)

cr_norm_knn_dist = ConformalRegressor()

cr_norm_knn_dist.fit(residuals_cal, sigmas=sigmas_cal_knn_dist)

print(cr_norm_knn_dist)


# To generate prediction intervals for the test set, we need difficulty estimates for the latter too, which we get in the same way as for the calibration objects. 

sigmas_test_knn_dist = de_knn.apply(X_test)

intervals_norm_knn_dist = cr_norm_knn_dist.predict(y_hat_test, 
                                                   sigmas=sigmas_test_knn_dist,
                                                   y_min=y_model_min, y_max=y_model_max)

def plot_pred_intervals(y_true, y_pred, intervals, title=''):
    
    # Prepare warnings when the prediction interval does not contain the true value
    warning1 = y_true > intervals[:,1]
    warning2 = y_true < intervals[:,0]
    warnings = warning1 + warning2
    error = abs(intervals[:,0]-intervals[:,1]) /2
    coverage = len(y_true[~warnings])/len(y_true)
    width = (intervals[:,1] - intervals[:,0])
    width_mean = width.mean()
    
    plt.plot(y_true, y_true, label="True values", color="black")
    plt.errorbar(y_true[~warnings], y_pred[~warnings], error[~warnings], alpha=0.5,fmt="o",color="blue",elinewidth=1,capsize=2,label="True Val In Prediction interval")
    plt.errorbar(y_true[warnings], y_pred[warnings], error[warnings], alpha=0.5,fmt="o",color="red",elinewidth=1,capsize=2,label="Prediction interval ERROR")

    plt.title(title+f"\nPrediction intervals with coverage {coverage:.4f} and width {width_mean:.4f}")
    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    
    plt.legend()
    plt.show()
    plt.title(title+f"\nPrediction intervals")
    #width_unscaled = width*true_max
    #plt.scatter(x=y_true*true_max,y=width_unscaled,marker="+")
    width_unscaled = width
    plt.scatter(x=y_true,y=width_unscaled,marker="+")
    plt.xlabel("True values")
    plt.ylabel("interval width")
    plt.show()


plot_pred_intervals(y_test,y_hat_test,intervals_norm_knn_dist,'norm_knn: test')

print(intervals_norm_knn_dist)

sigmas_cal_knn_dist = de_knn.apply(X_cal)

cal_intervals_norm_knn_dist = cr_norm_knn_dist.predict(y_hat_cal, 
                                                   sigmas=sigmas_cal_knn_dist,
                                                   y_min=y_model_min, y_max=y_model_max)

plot_pred_intervals(y_cal,y_hat_cal,cal_intervals_norm_knn_dist,'norm_knn: cal')
# Alternatively, we could estimate the difficulty using the standard deviation of the targets of the nearest neighbors; we specify this by providing the targets too:

de_knn_std = DifficultyEstimator(knn_target_strangeness=knn_targ_strange_flag)
#std difficulty estimate (have both x and y)
de_knn_std.fit(X=X_prop_train, y=y_prop_train, scaler=True)

print(de_knn_std)
#sigmas_cal_knn_std = de_knn_std.apply(X_cal)
if knn_targ_strange_flag:
    sigmas_cal_knn_std = de_knn_std.apply(X_cal, y_cal)
else:
    sigmas_cal_knn_std = de_knn_std.apply(X_cal)
    
cr_norm_knn_std = ConformalRegressor()

cr_norm_knn_std.fit(residuals_cal, sigmas=sigmas_cal_knn_std)

intervals_norm_knn_std = cr_norm_knn_std.predict(y_hat_cal, 
                                                 sigmas=sigmas_cal_knn_std,confidence=0.95,
                                                 y_min=y_model_min, y_max=y_model_max)
plot_pred_intervals(y_cal,y_hat_cal,intervals_norm_knn_std,'norm_knn_std: cal')
print(cr_norm_knn_std)


# ... and similarly for the test objects:

if knn_targ_strange_flag:
    sigmas_test_knn_std = de_knn_std.apply(X_test, y_test)
else:
    sigmas_test_knn_std = de_knn_std.apply(X_test)

intervals_norm_knn_std = cr_norm_knn_std.predict(y_hat_test, 
                                                 sigmas=sigmas_test_knn_std,confidence=0.95,
                                                 y_min=y_model_min, y_max=y_model_max)
plot_pred_intervals(y_test,y_hat_test,intervals_norm_knn_std,'norm_knn_std: test')
print(intervals_norm_knn_std)


# A third option is to use (absolute) residuals for the reference objects. For a model that overfits the training data, it can be a good idea to use a separate set of (reference) objects and labels from which the residuals could be calculated, rather than using the original training data. Since we in this case have trained a random forest, we opt for estimating the residuals by using the out-of-bag predictions for the training instances. (This was made possible by setting `oob_score=True` for the `RandomForestRegressor` above.)
# 
# To inform the `fit` method that this is what we want to do, 
#we provide a value for `residuals`, instead of `y` as we did above for the option to use the (standard deviation of) the targets.

oob_predictions = learner_prop.oob_prediction_

residuals_oob = y_prop_train - oob_predictions

de_knn_res = DifficultyEstimator(knn_target_strangeness=knn_targ_strange_flag)
#residual difficulty estimate
de_knn_res.fit(X=X_prop_train, residuals=residuals_oob, scaler=True)

print(de_knn_res)

sigmas_cal_knn_res = de_knn_res.apply(X_cal)

cr_norm_knn_res = ConformalRegressor()

cr_norm_knn_res.fit(residuals_cal, sigmas=sigmas_cal_knn_res)

intervals_norm_knn_res = cr_norm_knn_std.predict(y_hat_cal, 
                                                 sigmas=sigmas_cal_knn_res,
                                                 y_min=y_model_min, y_max=y_model_max)


plot_pred_intervals(y_cal,y_hat_cal,intervals_norm_knn_res,'norm_knn_res: cal')

print(cr_norm_knn_res)


# ... and again, the difficulty estimates are formed in the same way for the test objects:

sigmas_test_knn_res = de_knn_res.apply(X_test)

intervals_norm_knn_res = cr_norm_knn_res.predict(y_hat_test, 
                                                 sigmas=sigmas_test_knn_res,
                                                 y_min=y_model_min, y_max=y_model_max)

plot_pred_intervals(y_test,y_hat_test,intervals_norm_knn_res,'norm_knn_res: test')

print(intervals_norm_knn_res)


# In case we have trained an ensemble model, like a `RandomForestRegressor`, we could alternatively request `DifficultyEstimator` to estimate the difficulty by the variance of the predictions of the constituent models. This requires us to provide the trained model `learner` as input to `fit`, assuming that `learner.estimators_` is a collection of base models, each implementing the `predict` method; this holds e.g., for `RandomForestRegressor`. A set of objects (`X`) has to be provided only if we employ scaling (`scaler=True`).

de_var = DifficultyEstimator(knn_target_strangeness=knn_targ_strange_flag)

de_var.fit(X=X_prop_train, learner=learner_prop, scaler=True)

print(de_var)

sigmas_cal_var = de_var.apply(X_cal)

cr_norm_var = ConformalRegressor()

cr_norm_var.fit(residuals_cal, sigmas=sigmas_cal_var)

print(cr_norm_var)


# The difficulty estimates for the test set are generated in the same way:

sigmas_test_var = de_var.apply(X_test)

intervals_norm_var = cr_norm_var.predict(y_hat_test, 
                                         sigmas=sigmas_test_var, 
                                         y_min=y_model_min, y_max=y_model_max)

print(intervals_norm_var)


# ### Mondrian conformal regressors

# An alternative way of generating prediction intervals of varying size
# is to divide the object space into non-overlapping so-called Mondrian categories.
# A Mondrian conformal regressor is formed by providing the names of the categories
# as an additional argument, named `bins`, for the `fit` method.
# 
# Here we employ the helper function `binning`, imported from `crepes.extras`, which given a list/array of values
# returns an array of the same length with the assigned bins. 
# If the optional argument `bins` is an integer, the function will divide the values 
# into equal-sized bins and return both the assigned bins and the bin boundaries. 
# If `bins` instead is a set of bin boundaries, the function will just return the assigned bins.
# 
# We can form the Mondrian categories in any way we like, as long as we only
# use information that is available for both calibration and test instances;
# this means that we may not use the target values for this purpose, since these will 
# typically not be available for the test instances. 
# We will form categories by binning of the difficulty estimates, here using the
# ones previously produced using the standard deviations of the nearest neighbor targets. 

bins_cal, bin_thresholds = binning(sigmas_cal_var, bins=20)

cr_mond = ConformalRegressor()

cr_mond.fit(residuals_cal, bins=bins_cal)

print(cr_mond)


# Let us now obtain the categories for the test instances using the
# same Mondrian categorization, i.e., bin borders.

bins_test = binning(sigmas_test_var, bins=bin_thresholds)


# ... and now we can form prediction intervals for the test instances.

intervals_mond = cr_mond.predict(y_hat_test, bins=bins_test, y_min=y_model_min, y_max=y_model_max)

print(intervals_mond)


# ### Investigating the prediction intervals

# Let us first put all the intervals in a dictionary.

prediction_intervals = {
    "Std CR":intervals_std,
    "Norm CR knn dist":intervals_norm_knn_dist,
    "Norm CR knn std":intervals_norm_knn_std,
    "Norm CR knn res":intervals_norm_knn_res,
    "Norm CR var":intervals_norm_var,
    "Mond CR":intervals_mond,
}


# Let us see what fraction of the intervals that contain the true targets and how large the intervals are.

coverages = []
mean_sizes = []
std_sizes = []
median_sizes = []

for name in prediction_intervals.keys():
    intervals = prediction_intervals[name]
    coverages.append(np.sum([1 if (y_test[i]>=intervals[i,0] and 
                                   y_test[i]<=intervals[i,1]) else 0 
                            for i in range(len(y_test))])/len(y_test))
    mean_sizes.append((intervals[:,1]-intervals[:,0]).mean())
    std_sizes.append((intervals[:,1]-intervals[:,0]).std())
    median_sizes.append(np.median((intervals[:,1]-intervals[:,0])))

pred_no_oob_int_df = pd.DataFrame({"Coverage":coverages, 
                            "Mean size":mean_sizes, 
                            "Std size":std_sizes,
                            "Median size":median_sizes}, 
                           index=list(prediction_intervals.keys()))

pred_no_oob_int_df.loc["Mean"] = [pred_no_oob_int_df["Coverage"].mean(), 
                           pred_no_oob_int_df["Mean size"].mean(),
                           pred_no_oob_int_df["Std size"].mean(),
                           pred_no_oob_int_df["Median size"].mean()]

print(pred_no_oob_int_df.round(4))
input('Enter to Continue')

# Let us look at the distribution of the interval sizes.

interval_sizes = {}
for name in prediction_intervals.keys():
    interval_sizes[name] = prediction_intervals[name][:,1] \
    - prediction_intervals[name][:,0]

plt.figure(figsize=(6,6))
plt.ylabel("CDF")
plt.xlabel("Interval sizes")
plt.xlim(0,interval_sizes["Mond CR"].max()*1.25)

colors = ["b","r","g","y","k","m","c","orange"]

for i, name in enumerate(interval_sizes.keys()):
    if "Std" in name:
        style = "dotted"
    else:
        style = "solid"
    plt.plot(np.sort(interval_sizes[name]),
             [i/len(interval_sizes[name])
              for i in range(1,len(interval_sizes[name])+1)],
             linestyle=style, c=colors[i], label=name)

plt.legend()
plt.show()


# ### Evaluating the conformal regressors

# Let us put the six above conformal regressors in a dictionary, together with the corresponding difficulty estimates for the test instances (if any).

all_cr = {
    "Std CR": (cr_std, []),
    "Norm CR knn dist": (cr_norm_knn_dist, sigmas_test_knn_dist),
    "Norm CR knn std": (cr_norm_knn_std, sigmas_test_knn_std),
    "Norm CR knn res": (cr_norm_knn_res, sigmas_test_knn_res),
    "Norm CR var" : (cr_norm_var, sigmas_test_var),
    "Mond CR": (cr_mond, sigmas_test_var),
}


# Let us evaluate them using three confidence levels on the test set.
# We could specify a subset of the metrics to use by the named
# `metrics` argument of the `evaluate` method; here we use all, 
# which is the default.
# 
# Note that the arguments `sigmas` and `bins` can always be provided,
# but they will be ignored by conformal regressors not using them,
# e.g., both arguments will be ignored by the standard conformal regressors.

confidence_levels = [0.9,0.95,0.99]

names = list(all_cr.keys())

all_results = {}

for confidence in confidence_levels:
    for name in names:
        all_results[(name,confidence)] = all_cr[name][0].evaluate(
        y_hat_test, y_test, sigmas=all_cr[name][1],
        bins=bins_test, confidence=confidence, 
        y_min=y_model_min, y_max=y_model_max)

results_df = pd.DataFrame(columns=pd.MultiIndex.from_product(
    [names,confidence_levels]), index=list(list(
    all_results.values())[0].keys()))

for key in all_results.keys():
    results_df[key] = all_results[key].values()

print(results_df.round(4))


# ### Conformal regressors without a separate calibration set

# For conformal regressors that employ learners that use bagging, like random forests, we may consider an alternative strategy to dividing the original training set into a proper training and calibration set; we may use the out-of-bag (OOB) predictions, which allow us to use the full training set for both model building and calibration. It should be noted that this strategy does not come with the theoretical validity guarantee of the above (inductive) conformal regressors, due to that calibration and test instances are not handled in exactly the same way. In practice, however, conformal regressors based on out-of-bag predictions rarely do not meet the coverage requirements.

# #### Standard conformal regressors with out-of-bag calibration

# Let us first generate a model from the full training set and then get the residuals using the OOB predictions; we rely on that the learner has an attribute `oob_prediction_`, which e.g. is the case for a `RandomForestRegressor` if `oob_score` is set to `True` when created.

learner_full = RandomForestRegressor(n_jobs=-1, n_estimators=500, 
                                     oob_score=True)

learner_full.fit(X_train, y_train)


# Now we can obtain the residuals.

oob_predictions = learner_full.oob_prediction_

residuals_oob = y_train - oob_predictions


# We may now obtain a standard conformal regressor from these OOB residuals

# In[52]:


cr_std_oob = ConformalRegressor()

cr_std_oob.fit(residuals_oob)


# ... and apply it using the point predictions of the full model.

y_hat_full = learner_full.predict(X_test)

intervals_std_oob = cr_std_oob.predict(y_hat_full, y_min=y_model_min, y_max=y_model_max)

print(intervals_std_oob)


# #### Normalized conformal regressors with out-of-bag calibration

# We may also generate normalized conformal regressors from the OOB predictions. The `DifficultyEstimator` can be used also for this purpose; for the k-nearest neighbor approaches, the difficulty of each object in the training set will be computed using a leave-one-out procedure, while for the variance-based approach the out-of-bag predictions will be employed. 
# 
# By setting `oob=True`, we inform the `fit` method that we may request difficulty estimates for the provided set of objects; these will be retrieved by not providing any objects when calling the `apply` method.
# 
# Let us start with the k-nearest neighbor approach using distances only.

de_knn_dist_oob = DifficultyEstimator(knn_target_strangeness=knn_targ_strange_flag)

de_knn_dist_oob.fit(X=X_train, scaler=True, oob=True)

print(de_knn_dist_oob)

sigmas_knn_dist_oob = de_knn_dist_oob.apply()

cr_norm_knn_dist_oob = ConformalRegressor()

cr_norm_knn_dist_oob.fit(residuals_oob, sigmas=sigmas_knn_dist_oob)


# In order to apply the normalized OOB regressors to the test set, we
# need to generate difficulty estimates for the latter too.

sigmas_test_knn_dist_oob = de_knn_dist_oob.apply(X_test)

intervals_norm_knn_dist_oob = cr_norm_knn_dist_oob.predict(
    y_hat_full, sigmas=sigmas_test_knn_dist_oob, y_min=y_model_min, y_max=y_model_max)

print(intervals_norm_knn_dist_oob)


# For completeness, we will illustrate the use of out-of-bag calibration for the remaining approaches too. For k-nearest neighbors with labels, we do the following:

de_knn_std_oob = DifficultyEstimator(knn_target_strangeness=knn_targ_strange_flag)

de_knn_std_oob.fit(X=X_train, y=y_train, scaler=True, oob=True)

print(de_knn_std_oob)

sigmas_knn_std_oob = de_knn_std_oob.apply()

cr_norm_knn_std_oob = ConformalRegressor()

cr_norm_knn_std_oob.fit(residuals=residuals_oob, sigmas=sigmas_knn_std_oob)


if knn_targ_strange_flag == 1:
    sigmas_test_knn_std_oob = de_knn_std_oob.apply(X_test,y_test)
else:
    sigmas_test_knn_std_oob = de_knn_std_oob.apply(X_test)
   
intervals_norm_knn_std_oob = cr_norm_knn_std_oob.predict(
    y_hat_full, sigmas=sigmas_test_knn_std_oob, y_min=y_model_min, y_max=y_model_max)

print(intervals_norm_knn_std_oob)


# A third option is to use k-nearest neighbors with (OOB) residuals:

de_knn_res_oob = DifficultyEstimator(knn_target_strangeness=knn_targ_strange_flag)

de_knn_res_oob.fit(X=X_train, residuals=residuals_oob, scaler=True, oob=True)

print(de_knn_res_oob)

sigmas_knn_res_oob = de_knn_res_oob.apply()

cr_norm_knn_res_oob = ConformalRegressor()

cr_norm_knn_res_oob.fit(residuals_oob, sigmas=sigmas_knn_res_oob)

sigmas_test_knn_res_oob = de_knn_res_oob.apply(X_test)

intervals_norm_knn_res_oob = cr_norm_knn_res_oob.predict(
    y_hat_full, sigmas=sigmas_test_knn_res_oob, y_min=y_model_min, y_max=y_model_max)

print(intervals_norm_knn_res_oob)


# A fourth and final option for the normalized conformal regressors is to use variance as a difficulty estimate. We then leave labels and residuals out, but provide an (ensemble) learner. In contrast to when `oob=False`, we are here required to provide the (full) training set, from which the variance of the out-of-bag predictions will be computed. When applied to the test set, the full ensemble model will not be used to obtain the difficulty estimates, but instead a subset of the constituent models is used, following what could be seen as post hoc assignment of each test instance to a bag.      

de_var_oob = DifficultyEstimator(knn_target_strangeness=knn_targ_strange_flag)

de_var_oob.fit(X=X_train, learner=learner_full, scaler=True, oob=True)

print(de_var_oob)

sigmas_var_oob = de_var_oob.apply()

cr_norm_var_oob = ConformalRegressor()

cr_norm_var_oob.fit(residuals_oob, sigmas=sigmas_var_oob)

sigmas_test_var_oob = de_var_oob.apply(X_test)

intervals_norm_var_oob = cr_norm_var_oob.predict(y_hat_full, 
                                                 sigmas=sigmas_test_var_oob, 
                                                 y_min=y_model_min, y_max=y_model_max)

print(intervals_norm_var_oob)


# #### Mondrian conformal regressors with out-of-bag calibration

# We may form the categories using the difficulty estimates obtained from the OOB predictions. We here consider the difficulty estimates produced by the fourth above option (using variance) only. 

bins_oob, bin_thresholds_oob = binning(sigmas_var_oob, bins=20)

cr_mond_oob = ConformalRegressor()

cr_mond_oob.fit(residuals=residuals_oob, bins=bins_oob)


# ... and assign the categories for the test instances, ...


bins_test_oob = binning(sigmas_test_var_oob, bins=bin_thresholds_oob)


# ... and finally generate the prediction intervals.

intervals_mond_oob = cr_mond_oob.predict(y_hat_full,
                                         bins=bins_test_oob,
                                         y_min=y_model_min, y_max=y_model_max)

print(intervals_mond_oob)


# ### Investigating the OOB prediction intervals

prediction_intervals = {
    "Std CR OOB":intervals_std_oob,
    "Norm CR knn dist OOB":intervals_norm_knn_dist_oob,
    "Norm CR knn std OOB":intervals_norm_knn_std_oob,
    "Norm CR knn res OOB":intervals_norm_knn_res_oob,
    "Norm CR var OOB":intervals_norm_var_oob,
    "Mond CR OOB":intervals_mond_oob,
}


# Let us see what fraction of the intervals that contain the true targets and how large the intervals are.

coverages = []
mean_sizes = []
std_sizes = []
median_sizes = []

for name in prediction_intervals.keys():
    intervals = prediction_intervals[name]
    coverages.append(np.sum([1 if (y_test[i]>=intervals[i,0] and 
                                   y_test[i]<=intervals[i,1]) else 0 
                            for i in range(len(y_test))])/len(y_test))
    mean_sizes.append((intervals[:,1]-intervals[:,0]).mean())
    std_sizes.append((intervals[:,1]-intervals[:,0]).std())
    median_sizes.append(np.median((intervals[:,1]-intervals[:,0])))

pred_oob_int_df = pd.DataFrame({"Coverage":coverages, 
                            "Mean size":mean_sizes, 
                            "Std size":std_sizes,
                            "Median size":median_sizes}, 
                           index=list(prediction_intervals.keys()))

pred_oob_int_df.loc["Mean"] = [pred_oob_int_df["Coverage"].mean(), 
                           pred_oob_int_df["Mean size"].mean(),
                           pred_oob_int_df["Std size"].mean(),
                           pred_oob_int_df["Median size"].mean()]

print(pred_oob_int_df.round(4))
input('Enter to Continue')

# Let us look at the distribution of the interval sizes.

interval_sizes = {}
for name in prediction_intervals.keys():
    interval_sizes[name] = prediction_intervals[name][:,1] \
    - prediction_intervals[name][:,0]

plt.figure(figsize=(6,6))
plt.ylabel("CDF")
plt.xlabel("Interval sizes")
plt.xlim(0,interval_sizes["Mond CR OOB"].max()*1.25)

colors = ["b","r","g","y","k","m","c","orange"]

for i, name in enumerate(interval_sizes.keys()):
    if "Std" in name:
        style = "dotted"
    else:
        style = "solid"
    plt.plot(np.sort(interval_sizes[name]),
             [i/len(interval_sizes[name])
              for i in range(1,len(interval_sizes[name])+1)],
             linestyle=style, c=colors[i], label=name)

plt.legend()
plt.show()


# ## Conformal Predictive Systems (CPS)

# ### Creating and fitting CPS

# Let us create and fit standard and normalized conformal predictive systems, using the residuals from the calibration set (as obtained in the previous section), as well two conformal predictive systems using out-of-bag residuals; with and without normalization. As can be seen, the input for fitting conformal predictive systems is on the same format as for the conformal regressors.

cps_std = ConformalPredictiveSystem().fit(residuals_cal)

cps_norm = ConformalPredictiveSystem().fit(residuals_cal,
                                           sigmas=sigmas_cal_var)

cps_std_oob = ConformalPredictiveSystem().fit(residuals_oob)

cps_norm_oob = ConformalPredictiveSystem().fit(residuals_oob, 
                                               sigmas=sigmas_var_oob)


# Let us also create some Mondrian CPS, but in contrast to the Mondrian conformal regressors above, we here form the categories through binning of the predictions rather than binning of the difficulty estimates. We may use the latter, i.e., the sigmas, to obtain a normalized CPS for each category (bin).

bins_cal, bin_thresholds = binning(y_hat_cal, bins=5)

cps_mond_std = ConformalPredictiveSystem().fit(residuals_cal,
                                               bins=bins_cal)

cps_mond_norm = ConformalPredictiveSystem().fit(residuals_cal,
                                                sigmas=sigmas_cal_var,
                                                bins=bins_cal)


bins_oob, bin_thresholds_oob = binning(oob_predictions, bins=5)

cps_mond_std_oob = ConformalPredictiveSystem().fit(residuals_oob,
                                                   bins=bins_oob)

cps_mond_norm_oob = ConformalPredictiveSystem().fit(residuals_oob,
                                                    sigmas=sigmas_var_oob,
                                                    bins=bins_oob)


cps_norm_std = ConformalPredictiveSystem().fit(residuals_cal,
                                                sigmas=sigmas_cal_knn_std)

cps_mond_norm_std = ConformalPredictiveSystem().fit(residuals_cal,
                                                sigmas=sigmas_cal_knn_std,
                                                bins=bins_cal)

cps_mond_norm_res = ConformalPredictiveSystem().fit(residuals_cal,
                                                sigmas=sigmas_cal_knn_res,
                                                bins=bins_cal)


# ### Making predictions

# For the normalized approaches, we already have the difficulty estimates which are needed for the test instances. 
# For the Mondrian approaches, we also need to assign the new categories to the test instances.

bins_test = binning(y_hat_test, bins=bin_thresholds)

bins_test_oob = binning(y_hat_full, bins=bin_thresholds_oob)


# The output of the `predict` method of a `ConformalPredictiveSystem` will depend on how we specify the input. If we provide specific target values (using the parameter `y`), the method will output a p-value for each test instance, i.e., the probability that the true target is less than or equal to the provided values. The method assumes that either one value is provided for each test instance or that the same (single) value is provided for all test instances.
# 
# Here we will obtain the p-values from `cps_mond_norm` for the true targets of the test set:

p_values = cps_mond_norm.predict(y_hat_test,
                                 sigmas=sigmas_test_knn_res,
                                 bins=bins_test,
                                 y=y_test)

print(p_values)


# If we instead would like to get threshold values, with a specified probability that the true target is less than the threshold for each test instance, we may instead provide percentiles as input to the `predict` method. This is done through the parameter
# `lower_percentiles`, which denotes (one or more) percentiles for which a lower value
# will be selected in case a percentile lies between two values
# (similar to `interpolation="lower"` in `numpy.percentile`), or using
# `higher_percentiles`, which denotes (one or more) percentiles for which a higher value
# will be selected in such cases (similar to `interpolation="higher"` in `numpy.percentile`).
# 
# Here we will obtain the lowest values from `cps_mond_norm`, such that the probability for the target values being less than these is at least 50%:

thresholds = cps_mond_norm.predict(y_hat_test,
                                   sigmas=sigmas_test_knn_res,
                                   bins=bins_test,
                                   higher_percentiles=50)

print(thresholds)


# We can also specify both target values and percentiles; the resulting p-values will be returned in the first column, while any values corresponding to the lower percentiles will be included in the subsequent columns, followed by columns containing the values corresponding to the higher percentiles. The following call hence results in an array with five columns:

results = cps_mond_norm.predict(y_hat_test,
                                sigmas=sigmas_test_knn_res,
                                bins=bins_test,
                                y=y_test,
                                lower_percentiles=[2.5, 5],
                                higher_percentiles=[95, 97.5])

print(results)


# In addition to p-values and threshold values, we can request that the `predict` method returns the full conformal predictive distribution (CPD) for each test instance, as defined by the threshold values, by setting `return_cpds=True`. The format of the distributions vary with the type of conformal predictive system; for a standard and normalized CPS, the output is an array with a row for each test instance and a column for each calibration instance (residual), while for a Mondrian CPS, the default output is a vector containing one CPD per test instance (since the number of values may vary between categories). If the desired output instead is an array of distributions per category, where all distributions in a category have the same number of columns, which in turn depends on the number of calibration instances in the corresponding category, then `cpds_by_bins=True` may be specified. In case `return_cpds=True` is specified together with `y`, `lower_percentiles` or `higher_percentiles`, the output of `predict` will be a pair, with the first element holding the results of the above type and the second element will contain the CPDs. 
# 
# For the above Mondrian CPS, the following call to `predict` will result in a vector of distributions, with one element for each test instance.

cpds = cps_mond_norm.predict(y_hat_test,
                             sigmas=sigmas_test_knn_res,
                             bins=bins_test,
                             return_cpds=True)

print(f"No. of test instances: {len(y_hat_test)}")
print(f"Shape of cpds: {cpds.shape}")


# If we instead would prefer to represent these distributions by one array per category, we set `cpds_by_bins=True`, noting that it will be a bit trickier to associate a test instance to a specific distribution.  

cpds = cps_mond_norm.predict(y_hat_test,
                             sigmas=sigmas_test_knn_res,
                             bins=bins_test,
                             return_cpds=True, 
                             cpds_by_bins=True)

for i, cpd in enumerate(cpds):
    print(f"bin {i}: {cpd.shape[0]} test instances, {cpd.shape[1]} threshold values")

print(f"No. of test instances: {sum([c.shape[0] for c in cpds])}")


# We may also plot the conformal predictive distribution for some test object. In case the calibration set is very large, you may consider plotting an approximation of the full distribution by using a grid of values for `lower_percentiles` or `higher_percentiles`, instead of setting `return_cpds=True`. For the Mondrian CPS, the size of the calibration set for each bin is reasonable in this case, so we may just use the distributions directly.

cpds = cps_mond_norm_oob.predict(y_hat_full,
                                 bins=bins_test_oob,
                                 sigmas=sigmas_test_var_oob,
                                 return_cpds=True)

test_index = np.random.randint(len(y_hat_full)) # A test object is randomly selected
cpd = cpds[test_index]

p = np.array([i/len(cpd) for i in range(len(cpd))])

lower_index = np.where(p<=0.025)[0][-1]
mid_index = np.where(p>=0.50)[0][0]
upper_index = np.where(p>=0.975)[0][0]

low_percentile = cpd[lower_index]
median = cpd[mid_index]
high_percentile = cpd[upper_index]

plt.figure(figsize=(6,6))
plt.plot([y_hat_full[test_index],y_hat_full[test_index]],[0,1], color="tab:orange")
plt.plot([y_test[test_index],y_test[test_index]],[0,1], color="tab:red")
plt.xlabel("y")
plt.ylabel("Q(y)")
plt.ylim(0,1)

plt.plot([median,median],[0,1],"g--")
plt.plot([low_percentile,low_percentile],[0,1],"y--")
plt.legend(["ŷ","target","$y_{0.5}$","[$y_{0.025}$,$y_{0.975}$]"])
plt.plot([high_percentile,high_percentile],[0,1],"y--")
plt.plot(cpd,p, color="tab:blue")
rectangle = plt.Rectangle((low_percentile,0),
                          abs(high_percentile-low_percentile),1, color="y", 
                          alpha=0.05)
plt.gca().add_patch(rectangle)
plt.show()


# ### Analyzing the p-values

# Let us put all the generated CPS in a dictionary.
'''
all_cps = {"Std CPS":cps_std,
           "Std OOB CPS":cps_std_oob,
           "Norm CPS":cps_norm,
           "Norm OOB CPS":cps_norm_oob,
           "Mond CPS":cps_mond_std,
           "Mond OOB CPS":cps_mond_std_oob,
           "Mond norm CPS":cps_mond_norm,
           "Mond norm OOB CPS":cps_mond_norm_oob
          }
'''
if knn_targ_strange_flag == 1:
    all_cps = {"cps_norm_targ_strg":cps_norm_std,
           "cps_mond_norm_targ_strg":cps_mond_norm_std
           }
else:    
    all_cps = {"cps_norm_std":cps_norm_std,
           "cps_mond_norm_std":cps_mond_norm_std,
           "cps_norm_var": cps_norm,
           "cps_mond_norm_var":cps_mond_norm
           }           

# Now we will check if the p-values for the test targets seem to be uniformly distributed.

for i, name in enumerate(all_cps.keys()):

    if "OOB" in name:
        p_values = all_cps[name].predict(y_hat_full, 
                                         sigmas=sigmas_test_var_oob,
                                         bins=bins_test_oob, 
                                         y=y_test)
    elif "var" in name:
        p_values = all_cps[name].predict(y_hat_test, 
                                         sigmas=sigmas_test_var, 
                                         bins=bins_test, 
                                         y=y_test)
    else:
        p_values = all_cps[name].predict(y_hat_test, 
                                         sigmas=sigmas_test_knn_std,
                                         bins=bins_test, 
                                         y=y_test)

    #plt.subplot(len(all_cps.keys())//2,2,i+1)

    plt.scatter(np.sort(p_values),
                [(i+1)/len(y_test) for i in range(len(y_test))],
                label=name, c="y", marker=".", alpha=0.1)

    plt.plot([0,1],[0,1],"r--")
    plt.legend()
    plt.ylabel("fraction")
    plt.xlabel("p value")

plt.show()


# ### Investigating the coverage and size of extracted prediction intervals

# Let us investigate the extracted prediction intervals at the 95% confidence level. 
# This is done by a specifying percentiles corresponding to the interval endpoints.

all_cps_intervals = {}

coverages = []
mean_sizes = []
std_sizes = []
median_sizes = []

for idx, name in enumerate(all_cps.keys()):
    if "OOB" in name:
        intervals = all_cps[name].predict(y_hat_full, 
                                          sigmas=sigmas_test_var_oob, 
                                          bins=bins_test_oob,
                                          lower_percentiles=2.5, 
                                          higher_percentiles=97.5,
                                          y_min=y_model_min, y_max=y_model_max)
    elif "var" in name:
        intervals = all_cps[name].predict(y_hat_test, 
                                          sigmas=sigmas_test_var, 
                                          bins=bins_test,
                                          lower_percentiles=2.5, 
                                          higher_percentiles=97.5,
                                          y_min=y_model_min, y_max=y_model_max)
    else:
        intervals = all_cps[name].predict(y_hat_test, 
                                          sigmas=sigmas_test_knn_std,
                                          bins=bins_test,
                                          lower_percentiles=2.5, 
                                          higher_percentiles=97.5,
                                          y_min=y_model_min, y_max=y_model_max)
    all_cps_intervals[name] = intervals
    coverages.append(np.sum([1 if (y_test[i]>=intervals[i,0] and 
                                   y_test[i]<=intervals[i,1]) else 0
                            for i in range(len(y_test))])/len(y_test))
    mean_sizes.append((intervals[:,1]-intervals[:,0]).mean())
    std_sizes.append((intervals[:,1]-intervals[:,0]).std())
    median_sizes.append(np.median((intervals[:,1]-intervals[:,0])))

pred_int_df = pd.DataFrame({"Coverage":coverages, 
                            "Mean size":mean_sizes,
                            "Std size":std_sizes,
                            "Median size":median_sizes}, 
                           index=list(all_cps_intervals.keys()))

pred_int_df.loc["Mean"] = [pred_int_df["Coverage"].mean(), 
                           pred_int_df["Mean size"].mean(),
                           pred_int_df["Std size"].mean(),
                           pred_int_df["Median size"].mean()]

print(pred_int_df.round(4))
input('Enter to Continue')

# ### Investigating the distributions of extracted prediction intervals

# Let us take a look at the distribution of the interval sizes.

cps_interval_sizes = {}

for name in all_cps_intervals.keys():
    cps_interval_sizes[name] = \
    all_cps_intervals[name][:,1] - all_cps_intervals[name][:,0]

plt.figure(figsize=(6,6))
plt.ylabel("CDF")
plt.xlabel("Interval sizes")
#plt.xlim(0,cps_interval_sizes["Mond OOB CPS"].max()*1.25)

colors = ["b","r","g","y","k","m", "gray", "orange"]

for i, name in enumerate(cps_interval_sizes.keys()):
    if "Std" in name:
        style = "dotted"
    else:
        style = "solid"
    plt.plot(np.sort(cps_interval_sizes[name]),
             [i/len(cps_interval_sizes[name])
              for i in range(1,len(cps_interval_sizes[name])+1)],
             linestyle=style, c=colors[i], label=name)

plt.legend()
plt.show()


# ### Extracting medians

# Let us take a look at the medians; they can be derived using either lower or higher interpolation,
# but ideally the differences should be small.

all_cps_medians = {}

for name in all_cps.keys():
    if "OOB" in name:
        medians = all_cps[name].predict(y_hat_full, 
                                        sigmas=sigmas_test_var_oob, 
                                        bins=bins_test_oob,
                                        lower_percentiles=50, 
                                        higher_percentiles=50)
    elif "var" in name:
        medians = all_cps[name].predict(y_hat_test, 
                                        sigmas=sigmas_test_var, 
                                        bins=bins_test,
                                        lower_percentiles=50, 
                                        higher_percentiles=50)
    else:
        medians = all_cps[name].predict(y_hat_test, 
                                        sigmas=sigmas_test_knn_std, 
                                        bins=bins_test,
                                        lower_percentiles=50, 
                                        higher_percentiles=50)
    all_cps_medians[name] = medians
    print(name)
    print("\tMean difference of the medians:    {:.6f}".format((medians[:,1]-medians[:,0]).mean()))
    print("\tLargest difference of the medians: {:.6f}".format((medians[:,1]-medians[:,0]).max()))


# ### Another view of the medians and prediction intervals

sorted_prop_indexes = np.argsort(y_hat_test) 

sorted_full_indexes = np.argsort(y_hat_full) 

alpha=0.2

for i, name in enumerate(all_cps_intervals.keys()):

    #plt.subplot(len(all_cps_intervals.keys())//2,2,i+1)
    if "OOB" in name:
        indexes = sorted_full_indexes
        y_hat_ = y_hat_full
    else:
        indexes = sorted_prop_indexes
        y_hat_ = y_hat_test

    plt.title(name)
    plt.plot(y_hat_[indexes], all_cps_intervals[name][indexes,0], 
             color="r", alpha=alpha)
    plt.plot(y_hat_[indexes], all_cps_intervals[name][indexes,1], 
             color="r", alpha=alpha)
    plt.scatter(y_hat_[indexes],y_test[indexes],
                color="b", marker="o", alpha=alpha)
    plt.scatter(y_hat_[indexes],y_hat_[indexes],
                color="y", marker=".", alpha=alpha)
    plt.xlabel("predicted")
    plt.ylabel("endpoints")

    plt.show()


# ### Evaluating the CPS using a test set

# Let us evaluate the generated CPS using three confidence levels on the test set.
# We could specify a subset of the metrics to use by the
# `metrics` parameter of the `evaluate` method; here we use all metrics, 
# which is the default
# 
# Note that values for the parameters `sigmas` and `bins` can always be provided,
# but they will be ignored by CPS that have not been fitted with such values,
# e.g., both arguments will be ignored by the standard CPS.
# 
# Note that CRPS takes some time to compute, in particular when the CPS have been fitted with 
# larger calibration sets.

confidence_levels = [0.9,0.95,0.99]

names = np.array(list(all_cps.keys()))

first_set = names[["OOB" not in name for name in names]]
second_set = names[["OOB" in name for name in names]]

for methods in [names]:#[first_set]:#[first_set, second_set]:
    all_cps_results = {}
    for confidence in confidence_levels:
        for name in methods:
            if "OOB" in name:
                all_cps_results[(name,confidence)] = all_cps[name].evaluate(
                    y_hat_full, y_test, sigmas=sigmas_test_var_oob, 
                    bins=bins_test_oob, confidence=confidence, 
                    y_min=y_model_min, y_max=y_model_max)
            elif "var" in name:
                all_cps_results[(name,confidence)] =  all_cps[name].evaluate(
                    y_hat_test, y_test, sigmas=sigmas_test_var, 
                    bins=bins_test, confidence=confidence, y_min=y_model_min, y_max=y_model_max)
            else:
                all_cps_results[(name,confidence)] =  all_cps[name].evaluate(
                    y_hat_test, y_test, sigmas=sigmas_test_knn_std,
                    bins=bins_test, confidence=confidence, y_min=y_model_min, y_max=y_model_max)

    cps_results_df = pd.DataFrame(columns=pd.MultiIndex.from_product(
        [methods,confidence_levels]), index=list(list(
        all_cps_results.values())[0].keys()))

    for key in all_cps_results.keys():
        cps_results_df[key] = all_cps_results[key].values()

    print(cps_results_df.round(4))

