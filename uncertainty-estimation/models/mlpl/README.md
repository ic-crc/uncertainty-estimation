
The model generated is here for reference.

The model was used to generate the prediction column in the csv files in the /data/mlpl directory, as follows:

The MLPL code can be operated using the python code shown below:

# import python libraries (XGboost and Numpy)
from xgboost import XGBRegressor
import numpy as np

# load xgboost model from file
model_xg_load = XGBRegressor()
model_xg_load.load_model("UK_London_inputs3fold0.json")

# define input variables (these can be fixed or a list of inputs)
frequency = np.random.uniform(400, 6000)
link_distance = np.random.uniform(100, 30000)
obstacle_depth = np.random.uniform(0, link_distance/10)

# combine inputs into a 1x3 array
inputs = np.array([[frequency, link_distance, obstacle_depth]])
print("Input variables:",inputs)

# make prediction
PL_prediction = model_xg_load.predict(inputs)

# print prediction to console
print("Predicted Path Loss [dB]:",PL_prediction)    
