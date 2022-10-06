"""
 Sample script using EEGNet to classify Event-Related Potential (ERP) EEG data
 from a four-class classification task
   
 The four classes used from this dataset are:
     UP
     DOWN
     RIGHT
     LEFT
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
#import shutil

from Data_extractions import  Extract_data_from_subject
from Data_processing import  Select_time_window, Transform_for_classificator

from torch.autograd import Variable
from sklearn.preprocessing import RobustScaler


class EEGNet(torch.nn.Module):
    def __init__(self, n_output,sample_rate,F1,D,F2,n_channels,bias,dropout,activation,norm):
        super(EEGNet, self).__init__()

        # Block 1
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = F1, kernel_size=(1,sample_rate//2), padding= 0,bias=False)
        if norm == "BatchNorm":
            self.norm1 = nn.BatchNorm2d(F1, eps= 1e-05, momentum= 0.1, affine = False)

        # DepthwiseConv2D
        self.conv2 = nn.Conv2d(in_channels = F1, out_channels = F1*D, kernel_size=(n_channels,1), padding = 0,bias=False)
        if norm == "BatchNorm":
            self.norm2 = nn.BatchNorm2d(F1*D, eps= 1e-05, momentum= 0.1, affine =False)
        
        #Activation function
        if activation == "SILU":
            self.activation = nn.SiLU()
        elif activation == "ELU":
            self.activation = nn.ELU()
        elif activation == "GELU":
            self.activation = nn.GELU()
        
        self.pool1 = nn.AvgPool2d(kernel_size= (1,4))
        self.dropout1 = nn.Dropout(p = dropout)

        # Block 2
        # SeparableConv2D
        self.depthwise = nn.Conv2d(in_channels = F1*D, out_channels = F2, kernel_size=(1,sample_rate//8), padding=(0,16//2), groups = F2, bias=False) # Captures 500 ms of data at sampling rate 64Hz
        self.pointwise = nn.Conv2d(in_channels = F1*D , out_channels = F2, kernel_size = 1)
        if norm == "BatchNorm":
            self.norm3 = nn.BatchNorm2d(F2, eps=1e-05, momentum= 0.1, affine=False)
        
        if activation == "SILU":
            self.activation2 = nn.SiLU()
        elif activation == "ELU":
            self.activation2 = nn.ELU()
        elif activation == "GELU":
            self.activation2 = nn.GELU()
        
        self.pool2 = nn.AvgPool2d(kernel_size=(1,8))
        self.dropout2 = nn.Dropout(p= dropout)
        self.flatten = nn.Flatten()

        # Classifier
        if F1 == 16 and D ==4 :
            self.classifier = nn.Linear(1920, n_output,bias=bias)
        elif F1 == 24 and D==8:
            self.classifier = nn.Linear(5760, n_output, bias = bias)
        elif F1 == 32 and D==16:
            self.classifier = nn.Linear(15360, n_output, bias = bias)
        self.softmax = nn.LogSoftmax(dim = 1)

    def forward(self, x):
        # Block 1
        out = self.conv1(x)
        out = self.norm1(out)
        # DepthwiseConv2D
        out = self.conv2(out)
        out = torch.renorm(out , p=2, dim=0, maxnorm=1)
        out = self.norm2(out)
        out = self.activation(out)
        out = self.pool1(out)
        out = self.dropout1(out)
        # Block 2
        out = self.depthwise(out)
        out = self.pointwise(out)
        out = self.norm3(out)
        out = self.activation2(out)
        out = self.pool2(out)
        out = self.dropout2(out)
        out = self.flatten(out)
        # Classifier
        out = self.classifier(out)
        out = self.softmax(out)
        out = torch.renorm(out, p=2,dim =0, maxnorm = 0.25)

        return out


    def reset_parameters(self,bias):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if bias:
                    nn.init.constant_(m.bias, 0)


def testing(x_test,y_test,model,device,criterion,n_gpu):
    with torch.no_grad():
        model.cuda(n_gpu)
        n = x_test.shape[0]

        x_test = x_test.astype("float32")
        y_test = y_test.astype("float32").reshape(y_test.shape[0],)

        x_test, y_test = Variable(torch.from_numpy(x_test)),Variable(torch.from_numpy(y_test))
        y_test = torch.tensor(y_test,dtype=torch.long)
        
        x_test,y_test = x_test.to(device),y_test.to(device)
        y_pred_test = model(x_test)

        correct_test = (torch.max(y_pred_test,1)[1]==y_test).sum().item()
        test_accuracy = correct_test/n

        test_loss = criterion(y_pred_test,y_test).item()
        
    return test_accuracy,test_loss

def shuffle_in_unison(a, b):
    assert a.shape[0] == b.shape[0]
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(a.shape[0])
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index,:,:] = a[old_index,:,:]
        shuffled_b[new_index,] = b[old_index,]
    return shuffled_a, shuffled_b

def load_innerspeech_data(root_dir,datatype,N_S,t_start,t_end,sample_rate):
    # Load all trials for a single subject
    X, Y = Extract_data_from_subject(root_dir, N_S, datatype)

    # Cut usefull time. i.e action interval
    X = Select_time_window(X = X, t_start = t_start, t_end = t_end, fs = sample_rate)
    print("Data shape: [trials x channels x samples]")
    print(X.shape) # Trials, channels, samples

    print("Labels shape")
    print(Y.shape) # Time stamp, class , condition, session

    # Conditions to compared
    Conditions = [["Inner"],["Inner"],["Inner"],["Inner"]]
    # The class for the above condition
    Classes    = [  ["Up"] ,["Down"],["Right"],["Left"] ]
    # Transform data and keep only the trials of interes
    data , labels =  Transform_for_classificator(X, Y, Classes, Conditions)
    print("data shape InnerSpeech : ",data.shape)

    print("labels shape InnerSpeech : ",labels.shape)
    
    return data,labels

def save_model(model,save_dir,N_subj,N_fold):
    filename = save_dir + '/model_sub'+str(N_subj)+'_fold'+str(N_fold)
    f = open(filename,'w')
    torch.save(model.state_dict(), filename)
    
def load_model(model,save_dir,N_subj,N_fold):
    filename = save_dir + '/model_sub'+str(N_subj)+'_fold'+str(N_fold)
    model.load_state_dict(torch.load(filename))

def select_channels(data,channels_arr):
    count =0

    for channels in channels_arr:
        if(channels[0] == "A"):
            index = int(channels[1:])-1
        elif(channels[0] == "B"):
            index = 31 + int(channels[1:])
        elif(channels[0] == "C"):
            index = 63 + int(channels[1:])
        elif(channels[0] == "D"):
            index = 95 + int(channels[1:])

        #print("index : ",index)
        if count ==0:
            data1  = data[:,index,:]
            data_selec = data1.reshape((data1.shape[0],1,data1.shape[1]))
            #print("data_selec : ",data_selec.shape)
            count =1
        else:
            data1  = data[:,index,:]
            data_selec2 = data1.reshape((data1.shape[0],1,data1.shape[1]))
            data_selec = np.concatenate((data_selec,data_selec2),axis = 1)
    
    print("Size of the data after channels selection : ",data_selec.shape)
    return data_selec

def robust_scaler(data):
    #Center and scale the data
    print("Centering and scaling the data ...")
    for trials in range(data.shape[0]):
        transformer = RobustScaler().fit(data[trials,:,:])
        data_tr = transformer.transform(data[trials,:,:])
        data[trials,:,:] = data_tr
        data_tr = None
    return data



#Parameters
n_gpu = 2
datatype = "EEG"
n_class = 4
sample_rate = 256
t_start = 0
t_end = 4.5
select_chans = False
robustscaler = True
n_splits = 4
kernels, chans, samples = 1, 128, int((t_end - t_start)*sample_rate)
epochs = 50

norm = "BatchNorm"                       # LayerNorm/ BatchNorm
activation_arr = ["ELU","SILU","GELU"]
batch_size_arr = [4,10,20]

F1_D_arr = [[16,4],[24,8],[32,16]]
lr_arr = [0.1, 0.01, 0.001, 0.0001]
dropout_arr = [0.25,0.5]
bias_arr = [False,True]

 # Subjects
N_subj_arr = [6]

# Path of Preprocessed data
root_dir = '../Inner_Speech_Database'

if select_chans:
    channels_arr = ["A1","A2","A4","A5","A8","A9","A10","A15","A19","A20","A22","A23","A24","C18","C20","C27","C31","C32","D5","D6","D10","D15","D19","D21","D26","D30"]
    chans = len(channels_arr)

print("Number of channels selected : ",chans)

def classify_K_fold(N_S,data, labels, activation, batch_size, lr, dropout, bias, F1, D):
    # Path to save scores and models
    save_dir = './All_exp/'+str(activation)+'_dropout='+str(dropout)+'_F1='+str(F1)+'_D='+str(D)+'_lr='+str(lr)+'_bs='+str(batch_size)+'_bias='+str(bias)+'_'+str(chans)+'chans_NLLLoss'

    #K-fold cross validation
    data2, labels2 = shuffle_in_unison(data,labels)

    data_fold_arr = np.split(data2,n_splits,axis = 0)
    labels_fold_arr = np.split(labels2, n_splits, axis = 0)

    for offset in range(1,n_splits+1):
        print("Test fold number ",n_splits-offset+1)
        if offset >1:
            model.reset_parameters(bias)
            
        X_test = data_fold_arr[n_splits-offset]
        Y_test = labels_fold_arr[n_splits-offset]
        X_train_arr = [x for i,x in enumerate(data_fold_arr) if i!=n_splits-offset]
        Y_train_arr = [x for i,x in enumerate(labels_fold_arr) if i!=n_splits-offset]

        X_train, Y_train = np.vstack(X_train_arr), np.hstack(Y_train_arr)

        n = X_train.shape[0]

        print("X_train_shape : ",X_train.shape)
        print("Y_train_shape :",Y_train.shape)
        print("X_test_shape : ",X_test.shape)
        print("Y_test_shape :",Y_test.shape)

        ############################# EEGNet portion ##################################

        # convert data to NHWC (trials, kernels, channels, samples) format. Data 
        # contains 128 channels and 1152 time-points. Set the number of kernels to 1.
        X_train      = X_train.reshape(X_train.shape[0],kernels, chans, samples)
        X_test       = X_test.reshape(X_test.shape[0],kernels, chans, samples)

        # print('X_train shape:', X_train.shape)
        # print('Y_train shape:', Y_train.shape)
        # print(X_train.shape[0], 'train samples')
        # print(X_test.shape[0], 'test samples')

        X_train = X_train.astype("float32")
        Y_train = Y_train.astype("float32").reshape(Y_train.shape[0],)

        # train_data.shape = (essai,1,canaux,samples)

        X_train, Y_train = Variable(torch.from_numpy(X_train)),Variable(torch.from_numpy(Y_train))
        Y_train = torch.tensor(Y_train, dtype=torch.long) 


        model = EEGNet(n_output=n_class,sample_rate = sample_rate,F1 =F1,D=D,F2=F1*D,n_channels = chans, bias = bias, dropout = dropout, activation = activation, norm = norm)
        print(model)

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(),lr = lr)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,300,500], gamma=0.1)

        model.cuda(n_gpu)

        test_loss_history = []
        train_loss_history = []
        train_accuracy_history = []
        test_accuracy_history = []
        device = torch.device("cuda:"+str(n_gpu) if torch.cuda.is_available() else "cpu")
        X_train,Y_train = X_train.to(device),Y_train.to(device)

        for epoch in range(epochs):
            model.train()
                
            epoch_train_loss = 0
            for i in range(X_train.shape[0]//batch_size):
                start = i*batch_size
                end = (i+1)*batch_size

                data_train = X_train[start:end,:,:]
                labels_train = Y_train[start:end,]

                # print("data_train_batch : " ,data_train.shape)
                # print("labels_train_batch : " ,labels_train.shape)
                Y_pred = model(data_train)

                loss = criterion(Y_pred, labels_train)
                epoch_train_loss+= loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            model.eval()

            # Training evaluation
            train_loss = epoch_train_loss.item()/(X_train.shape[0]//batch_size)
            train_loss_history.append(train_loss)

            n = labels_train.shape[0]
            correct = (torch.max(Y_pred,1)[1]==labels_train).sum().item()
            train_accuracy = correct / n
            train_accuracy_history.append(train_accuracy)

                
            # Testing evaluation
            test_accuracy,test_loss = testing(X_test,Y_test,model,device,criterion,n_gpu)
            test_accuracy_history.append(test_accuracy)
            test_loss_history.append(test_loss)

            print("epochs:",epoch+1,"Test Loss",test_loss,"Train Loss",train_loss,"Training Accuracy:",train_accuracy,"Testing Accuracy:",test_accuracy,"Learning rate:",scheduler.get_last_lr()[0])

        df = pd.DataFrame({"train_loss_history":train_loss_history,"test_loss_history":test_loss_history,"train_accuracy_history":train_accuracy_history,"test_accuracy_history":test_accuracy_history})

        file_path_csv = save_dir + '/Scores_sub_'+str(N_S)+'_fold_'+str(n_splits-offset+1)+'.csv'
        df.to_csv(file_path_csv,encoding="utf-8-sig")

        #save_model(model,save_dir,N_S,n_splits-offset+1)


# Test all parameters combinations
n_tot_exp = len(activation_arr)*len(batch_size_arr)*len(lr_arr)*len(dropout_arr)*len(bias_arr)*len(F1_D_arr)*len(N_subj_arr)
count_exp = 0

for N_S in N_subj_arr:
    print('Subject: ' + str (N_S))

    # Load data for a subject
    data, labels = load_innerspeech_data(root_dir,datatype,N_S,t_start,t_end,sample_rate)
    if select_chans:   
        data_selec = select_channels(data, channels_arr)
        if robustscaler:
            data_selec = robust_scaler(data_selec)
    else:
        if robustscaler:
            data_selec = robust_scaler(data)
        else:
            data_selec = data

    for activation in activation_arr:
        for batch_size in batch_size_arr:
            for lr in lr_arr:
                for dropout in dropout_arr:
                    for bias in bias_arr:
                        for F1_D in F1_D_arr:
                            F1 = F1_D[0]
                            D = F1_D[1]

                            print("activation:",activation,"batch_size:",batch_size,"lr:",lr,"dropout:",dropout,"bias:",bias,"F1:",F1,"D:",D)
                            count_exp +=1
                            print("Experience : "+ str(count_exp)+"/"+str(n_tot_exp))
                            print("\n")
                            
                            classify_K_fold(N_S,data_selec, labels, activation, batch_size, lr, dropout, bias, F1, D)

