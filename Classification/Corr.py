"""
 Sample script computing a cross-correlation coefficient between two subjects using ReliefF and Pearson correlation
 coefficient
"""

import numpy as np

from ReliefF import reliefF
from sklearn.preprocessing import RobustScaler
from Data_extractions import  Extract_data_from_subject
from Data_processing import  Select_time_window, Transform_for_classificator, Split_trial_in_time

def load_innerspeech_data(root_dir,datatype,N_S,t_start,t_end,sample_rate,robust_scaler,Class):
    # Load all trials for a single subject
    X, Y = Extract_data_from_subject(root_dir, N_S, datatype)

    # Cut usefull time. i.e action interval
    X = Select_time_window(X = X, t_start = t_start, t_end = t_end, fs = sample_rate)
    #print("Data shape: [trials x channels x samples]")
    #print(X.shape) # Trials, channels, samples

    #print("Labels shape")
    #print(Y.shape) # Time stamp, class , condition, session

    # Conditions to compared
    Conditions = [["Inner"]]
    # The class for the above condition
    Classes    = [[Class]]
    # Transform data and keep only the trials of interes
    data , labels =  Transform_for_classificator(X, Y, Classes, Conditions)
    #print("data shape InnerSpeech : ",data.shape)

    #print("labels shape InnerSpeech : ",labels.shape)

    #Center and scale the data
    if robust_scaler:
        print("Centering and scaling the data ...")
        for trials in range(data.shape[0]):
            transformer = RobustScaler().fit(data[trials,:,:])
            data_tr = transformer.transform(data[trials,:,:])
            data[trials,:,:] = data_tr
            data_tr = None
    
    return data,labels

#Compute the correlation coeffient 
# If you are interested in the computed weights, print them with
#print(fs.w_) .Each i-th weight will be the weight of the i-th feature
def Pearson_corr_coeff(data_source,data_target,labels_source,labels_target):
    print("Computing Pearson correlation coefficient ...")
    n_trial_source = data_source.shape[0]
    n_trial_target = data_target.shape[0]

    #contain the highest-ranked feature vectors from data
    for trial_source in range(n_trial_source):
        #print("essai source numero :",trial_source)
        if trial_source == 0:
            score_feature_source = reliefF(data_source[trial_source,:,:], labels_source)
        else:
            score_feature_source= score_feature_source + reliefF(data_source[trial_source,:,:], labels_source)

        print("score_feature_source size : ",score_feature_source.shape)
        
    for trial_target in range(n_trial_target):
        #print("essai target numero :",trial_target)
        if trial_target == 0:
            score_feature_target = reliefF(data_target[trial_target,:,:], labels_target)
        else:
            score_feature_target= score_feature_target + reliefF(data_target[trial_target,:,:], labels_target)
        

    score_feature_source = score_feature_source/n_trial_source
    score_feature_target = score_feature_target/n_trial_target

    corr_mat = np.corrcoef(score_feature_source,score_feature_target)

    mat_len = corr_mat.shape[0]
    corr_coeff = 0
    for i in range(mat_len):
        for j in range(mat_len):
            if i!=j:
                corr_coeff = corr_coeff + corr_mat[i,j]
    corr_coeff = corr_coeff / (mat_len*mat_len-mat_len)

    return corr_coeff

#Parameters
sample_rate = 256
datatype =  "EEG"
t_start = 0
t_end = 4.5
robust_scaler = True

# Subjects
N_subj_source_arr = [1,2,3,4,5,6,7,8,9,10]
N_subj_target_arr = [1,2,3,4,5,6,7,8,9,10]
Class_arr = ["Up","Down","Right","Left"]
channels = 128
samples = 1152

# Path of Preprocessed data
root_dir = '../Inner_Speech_Database'

## Computing the correlation matrix for all the subjects
corr_mat_sub = np.eye(len(N_subj_source_arr),len(N_subj_source_arr))

for N_subj_target in N_subj_target_arr:
    print("Target subject : ",N_subj_target)
    data_target_up, labels_target_up = load_innerspeech_data(root_dir,datatype,N_subj_target,t_start,t_end,sample_rate,robust_scaler,"Up")
    data_target_down, labels_target_down = load_innerspeech_data(root_dir,datatype,N_subj_target,t_start,t_end,sample_rate,robust_scaler,"Down")
    data_target_right, labels_target_right = load_innerspeech_data(root_dir,datatype,N_subj_target,t_start,t_end,sample_rate,robust_scaler,"Right")
    data_target_left, labels_target_left = load_innerspeech_data(root_dir,datatype,N_subj_target,t_start,t_end,sample_rate,robust_scaler,"Left")

    for N_S in [x for i,x in enumerate(N_subj_source_arr) if i!=N_subj_target-1]:
        print("Subject test : ",N_S)

        corr_coeff = 0
        for Class in Class_arr:
            #print("Class : ",Class)
            if Class == "Up":
                data_target = data_target_up
            elif Class == "Down":
                data_target = data_target_down
            elif Class == "Right":
                data_target = data_target_right
            elif Class == "Left":
                data_target = data_target_left
            
            data, labels = load_innerspeech_data(root_dir,datatype,N_S,t_start,t_end,sample_rate,robust_scaler,Class)

            corr_mat_class = np.eye(data_target.shape[0],data.shape[0])

            print("Computing corr_coeff for class ",Class)
            for M in range(data_target.shape[0]):
                for N in range(data.shape[0]):
                    pearson_corr_arr = np.zeros(channels)
                    for chans in range(channels):
                        corr_mat = np.corrcoef(data_target[M,chans,:],data[N,chans,:])
                        pearson_corr_arr[chans] = corr_mat[0,1]
                    #print(pearson_corr_arr)
                    chans_avr = np.mean(pearson_corr_arr)
                    #print("pearson corr for "+str(M)+","+str(N)+" : ",chans_avr)
                    corr_mat_class[M,N] = chans_avr
            
            #print("corr_mat_class : ",corr_mat_class)
            corr_coeff_class  = np.average(corr_mat_class)
            #print("corr_coeff for class "+str(Class)+" : ",corr_coeff_class)

            corr_coeff += corr_coeff_class

        corr_coeff = corr_coeff/len(Class_arr)
        print("Pearson correlation coeff with source subject "+str(N_S)+" :",corr_coeff)
        corr_mat_sub[N_subj_target-1,N_S-1] = corr_coeff

print("corr_mat_sub : ",corr_mat_sub)

# Save the correlation coeffcient subject matrix in a txt
with open("corr_mat_sub.txt","w+") as file:
  content = str(corr_mat_sub)
  file.write(content)

corr_mat_sub = np.array([[1,         0.00829636 ,0.0068064  ,0.00742791 ,0.00699218, 0.00439705,0.00287336 ,0.00456059 ,0.00665099 ,0.0038881 ]
 ,[0.00829636, 1      ,   0.00451569, 0.00380707 ,0.00520748, 0.00293643,0.00182435 ,0.00440721, 0.00453122, 0.00229852]
 ,[0.0068064 , 0.00451569, 1        , 0.00701482, 0.00393642, 0.0057163,0.00281696, 0.00612717, 0.00675805, 0.00408275]
 ,[0.00742791, 0.00380707, 0.00701482, 1       ,  0.00733391, 0.00617534,0.00400834 ,0.00416469, 0.00571284 ,0.00433884]
 ,[0.00699218, 0.00520748, 0.00393642, 0.00733391 ,1         ,0.00512697,0.00223562 ,0.00348622, 0.00559036 ,0.00297022]
 ,[0.00439705, 0.00293643, 0.0057163 , 0.00617534, 0.00512697, 1,0.00342018, 0.00353229, 0.00616145, 0.00431859]
 ,[0.00287336 ,0.00182435 ,0.00281696, 0.00400834, 0.00223562, 0.00342018,1        , 0.00167385, 0.00277766, 0.00487118]
 ,[0.00456059 ,0.00440721 ,0.00612717 ,0.00416469, 0.00348622, 0.00353229,0.00167385 ,1         ,0.00455293, 0.0031456 ]
 ,[0.00665099, 0.00453122, 0.00675805 ,0.00571284, 0.00559036 ,0.00616145,0.00277766, 0.00455293, 1        , 0.00411098]
 ,[0.0038881 , 0.00229852, 0.00408275 ,0.00433884 ,0.00297022, 0.00431859,0.00487118, 0.0031456 , 0.00411098, 1        ]])

# Source selection
for N_subj_target in N_subj_target_arr:
    treshold = 0
    N_subj_selec=[]

    # Compute the treshold to select subject for TL pretraining (mean of all corr_coeff)
    for N_subj_source in N_subj_source_arr:
        if N_subj_source !=N_subj_target :
            treshold += corr_mat_sub[N_subj_target-1,N_subj_source-1]
    treshold = treshold / (len(N_subj_source_arr)-1)

    # Select subjects more correlated
    for N_subj_source in N_subj_source_arr:
        if N_subj_source !=N_subj_target :
            if corr_mat_sub[N_subj_target-1,N_subj_source-1]>= treshold:
                N_subj_selec.append(N_subj_source)

    print("Treshold for target sub "+str(N_subj_target)+" : ",treshold)
    print("Subject selectioned for target sub "+str(N_subj_target)+" : ",N_subj_selec)








# for N_subj_target in N_subj_target_arr:
#     print("Target subject : ",N_subj_target)
#     data_target,labels_target = load_innerspeech_data2(root_dir,datatype,N_subj_target,t_start,t_end,sample_rate,robust_scaler)
    
#     #Source selection
#     N_subj_selec=[]
#     for N_S in [x for i,x in enumerate(N_subj_source_arr) if i!=N_subj_target-1]:
#         print("Subject test : ",N_S)

#         data_source,labels_source = load_innerspeech_data2(root_dir,datatype,N_S,t_start,t_end,sample_rate,robust_scaler)
#         corr_coeff = Pearson_corr_coeff(data_source,data_target,labels_source,labels_target)
