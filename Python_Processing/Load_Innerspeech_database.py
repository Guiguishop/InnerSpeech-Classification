import mne 
import warnings
import numpy as np



from Data_extractions import  Extract_data_from_subject
from Data_processing import  Select_time_window, Transform_for_classificator, Split_trial_in_time

np.random.seed(23)

mne.set_log_level(verbose='warning') #to avoid info at terminal
warnings.filterwarnings(action = "ignore", category = DeprecationWarning ) 
warnings.filterwarnings(action = "ignore", category = FutureWarning )


### Hyperparameters
# The root dir have to point to the folder that cointains the database
root_dir = "../../Inner_Speech_Database"

# Data Type
datatype = "EEG"

# Sampling rate
fs = 256

# Select the useful par of each trial. Time in seconds
t_start = 0
t_end = 4.5

# Subject number
N_S = 1   #[1 to 10]

# Load all trials for a single subject
X, Y = Extract_data_from_subject(root_dir, N_S, datatype)

# Cut usefull time. i.e action interval
X = Select_time_window(X = X, t_start = t_start, t_end = t_end, fs = fs)
print("Data shape: [trials x channels x samples]")
print(X.shape) # Trials, channels, samples

print("Labels shape")
print(Y.shape) # Time stamp, class , condition, session

# Conditions to compared
Conditions = [["Inner"],["Inner"],["Inner"],["Inner"]]
# The class for the above condition
Classes    = [  ["Up"] ,["Down"],["Right"],["Left"] ]
# Transform data and keep only the trials of interes
X , Y =  Transform_for_classificator(X, Y, Classes, Conditions)
print("Final data shape")
print(X.shape)

print("Final labels shape")
print(Y.shape)