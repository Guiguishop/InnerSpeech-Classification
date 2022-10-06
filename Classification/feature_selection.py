"""
128 channel EEG signals of each trial were
first transformed to the frequency domain using short-time
Fourier transform with 50%-overlapping 1-s Hamming window.
After applying a band-pass filter with a frequency range of
1–50 Hz, the estimated spectral time series of each channel
was then grouped into five stereotyped frequency bands,
including δ(1–3 Hz), θ(4–7 Hz), α(8–13 Hz), β(14–30 Hz)
and γ(31–50 Hz). This study then adopted a feature type of
differential laterality (DLAT; Lin et al., 2014) to reflect EEG
spectral dynamics of emotional responses in a representation of
hemispheric spectral asymmetry. Given 12 left-right symmetric
channels (available in a 30-channel montage) and five frequency
bands, DLAT generated a feature dimension of 60. Each
spectral time series of DLAT was further divided by the mean
power of its first 5 s for each trial followed by the gain
model-based calibration method (Grandchamp and Delorme,
2011). Afterwards, the DLAT features were z-transformed
across 16 trials to zero mean and unit variance for each
subject.
Rather than utilizing the entire DLAT space, this study
adopted a well-known feature selection method namely ReliefF
(Robnik-Šikonja and Kononenko, 2003) to exploit a minimal
yet optimal set of most informative features for each subject,
which has been demonstrated effective in Jenke et al. (2014).
The number of features with high ReliefF weight was
determined based on the best training accuracy (described
later).


"""

import numpy as np
import scipy
import matplotlib.pyplot as plt

from Data_extractions import  Extract_data_from_subject
from Data_processing import  Select_time_window, Transform_for_classificator
from ReliefF import ReliefF

from sklearn.preprocessing import RobustScaler
from scipy import signal

def load_innerspeech_data(root_dir,datatype,N_S,t_start,t_end,sample_rate,robust_scaler):
    # Load all trials for a single subject
    X, Y = Extract_data_from_subject(root_dir, N_S, datatype)

    # Cut usefull time. i.e action interval
    X = Select_time_window(X = X, t_start = t_start, t_end = t_end, fs = sample_rate)
    # print("Data shape: [trials x channels x samples]")
    # print(X.shape) # Trials, channels, samples

    # print("Labels shape")
    # print(Y.shape) # Time stamp, class , condition, session

    # Conditions to compared
    Conditions = [["Inner"],["Inner"],["Inner"],["Inner"]]
    # The class for the above condition
    Classes    = [  ["Up"] ,["Down"],["Right"],["Left"] ]
    # Transform data and keep only the trials of interes
    data , labels =  Transform_for_classificator(X, Y, Classes, Conditions)
    # print("data shape InnerSpeech : ",data.shape)

    # print("labels shape InnerSpeech : ",labels.shape)

    #Center and scale the data
    if robust_scaler:
        #print("Centering and scaling the data ...")
        for trials in range(data.shape[0]):
            transformer = RobustScaler().fit(data[trials,:,:])
            data_tr = transformer.transform(data[trials,:,:])
            data[trials,:,:] = data_tr
            data_tr = None
    
    return data,labels

def bandPassFilter(signal,sample_rate):
    fs = sample_rate
    lowcut = 1
    highcut = 50

    nyq = 0.5*fs
    low = lowcut/nyq
    high = highcut/nyq

    order = 2

    b,a = scipy.signal.butter(order, [low,high],'bandpass',analog=False)
    y = scipy.signal.filtfilt(b,a,signal,axis= -1)

    return y

#Parameters
datatype = "EEG"
sample_rate = 256
t_start = 0
t_end = 4.5
robust_scaler = False
kernels, chans, samples = 1, 128, int((t_end - t_start)*sample_rate)


# Subjects
N_subj_source_arr = [1,2,3,4,5,6,7,8,9,10]
N_subj_target_arr = [1,2,3,4,5,6,7,8,9,10]
trials_min = 0

# Path of Preprocessed data
root_dir = '../Inner_Speech_Database'

def extract_features(N_S,root_dir,datatype,t_start,t_end,sample_rate,robust_scaler,samples,chans):
    data, labels = load_innerspeech_data(root_dir,datatype,N_S,t_start,t_end,sample_rate,robust_scaler)
    trials = data.shape[0]

    # Filtering between 1-50 Hz
    data_filtered = bandPassFilter(data,sample_rate)

    # Compute PSD for each trial and each channels with an hamming window, 50 % overlapping
    f, PSD = scipy.signal.welch(data_filtered, fs=sample_rate, window='hamming', nperseg=samples//6, noverlap=samples//12, axis=- 1)

    #print("PSD shape : ",PSD.shape) # -> (200,128,97) = (trials,channels,frequency)
    #print("frequencies :",f)
    # for trial in range(PSD.shape[0]):
    #     plt.plot(f,PSD[trial,1,:])
    #     plt.title("PSD for channel 0 and trial "+str(trial))
    #     plt.show()
    
    # five stereotyped frequency bands, including δ(1–3 Hz), θ(4–7 Hz), α(8–13 Hz), β(14–30 Hz) and γ(31–50 Hz)
    all_features_matrix = np.zeros((5*trials,chans))
    
    for trial in range(trials):
        for channel in range(chans):
            PSD_delta = np.max(PSD[trial,channel,1:2])
            PSD_theta = np.max(PSD[trial,channel,3:5])
            PSD_alpha = np.max(PSD[trial,channel,6:9])
            PSD_beta = np.max(PSD[trial,channel,11:22])
            PSD_gamma = np.max(PSD[trial,channel,24:37])
            all_features_matrix[trial*5+0,channel] = PSD_delta
            all_features_matrix[trial*5+1,channel] = PSD_theta
            all_features_matrix[trial*5+2,channel] = PSD_alpha
            all_features_matrix[trial*5+3,channel] = PSD_beta
            all_features_matrix[trial*5+4,channel] = PSD_gamma

    #print("feature matrix : ",all_features_matrix)

    # Z-transform (zero mean and unit variance across each trial)
    for row in range(all_features_matrix.shape[0]):
        mean = np.mean(all_features_matrix[row,:])
        std = np.std(all_features_matrix[row,:])
        all_features_matrix[row,:] = (all_features_matrix[row,:]-mean)/std

    #print(all_features_matrix)

    # ReliefF feature selection
    X = np.zeros((trials,5*chans))
    Y = labels

    for trial in range(trials):
        count = 0
        for i in range(5):
            for j in range(chans):
                X[trial,count] = all_features_matrix[5*trial+i,j]
                count+=1
                # print("index 1:",5*trial+i,"index2:",j)
                # print("trial :",trial,"count :",count)
    

    print("X shape : ",X.shape)
    fs = ReliefF(n_neighbors=100, n_features_to_keep= int(0.9*5*chans))
    final_features = fs.fit_transform(X,Y)

    print("final features shape :",final_features.shape)
    print(final_features)
    
    return final_features


for N_subj_target in N_subj_target_arr:
    #print("Subject target : ",N_subj_target)
    features_target = extract_features(N_subj_target,root_dir,datatype,t_start,t_end,sample_rate,robust_scaler,samples,chans)

    #print(features_target.shape)

    for N_S in [x for i,x in enumerate(N_subj_source_arr) if i!=N_subj_target-1]:
        #print("Subject source : ",N_S)
        features_source = extract_features(N_S,root_dir,datatype,t_start,t_end,sample_rate,robust_scaler,samples,chans)
        #print(features_source.shape)

        corr_mat = np.corrcoef(features_target,features_source, rowvar = True)
        #print("corr_matrix shape :",corr_mat.shape)

        corr_coeff = 0
        mat_len = corr_mat.shape[0]
        for i in range(mat_len):
            for j in range(mat_len):
                if i!=j:
                    corr_coeff = corr_coeff + corr_mat[i,j]
        corr_coeff = corr_coeff / (mat_len*mat_len-mat_len)

        print("Corr coeff between subject "+str(N_subj_target)+" and subject "+str(N_S)+":",corr_coeff)


    # Save the feature subject vector in a txt
    # with open("feature_sub"+str(N_S)+".txt","w+") as file:
    #     content = str(final_features)
    #     file.write(content)
