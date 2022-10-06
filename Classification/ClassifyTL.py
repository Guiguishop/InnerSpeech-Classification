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
import shutil

from Data_extractions import  Extract_data_from_subject
from Data_processing import  Select_time_window, Transform_for_classificator

from torch.autograd import Variable
from sklearn.preprocessing import RobustScaler

class EEGNet(torch.nn.Module):
    def __init__(self, n_output,sample_rate,F1,D,F2,n_channels,bias,dropout,activation):
        super(EEGNet, self).__init__()

        # Block 1
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = F1, kernel_size=(1,sample_rate//2), padding= 0,bias=False)
        self.norm1 = nn.BatchNorm2d(F1, eps= 1e-05, momentum= 0.1, affine = False)

        # DepthwiseConv2D
        self.conv2 = nn.Conv2d(in_channels = F1, out_channels = F1*D, kernel_size=(n_channels,1), padding = 0,bias=False)
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

def load_innerspeech_data(root_dir,datatype,N_S,t_start,t_end,sample_rate,robust_scaler):
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

    #Center and scale the data
    if robust_scaler:
        print("Centering and scaling the data ...")
        for trials in range(data.shape[0]):
            transformer = RobustScaler().fit(data[trials,:,:])
            data_tr = transformer.transform(data[trials,:,:])
            data[trials,:,:] = data_tr
            data_tr = None
    
    return data,labels

def save_model(model,save_dir,N_subj,N_fold):
    filename = save_dir + '/model_sub'+str(N_subj)+'_fold'+str(N_fold)
    f = open(filename,'w')
    torch.save(model.state_dict(), filename)

def save_model_pretrained(model,save_dir,N_subj_target):
    filename = save_dir + '/model_pretrained_sub'+str(N_subj_target)
    f = open(filename,'w')
    torch.save(model.state_dict(),filename)
    
def load_model(model,save_dir,N_subj,N_fold):
    filename = save_dir + '/model_sub'+str(N_subj)+'_fold'+str(N_fold)
    model.load_state_dict(torch.load(filename))

def load_model_pretrained(model,save_dir,N_subj_target):
    filename = save_dir + '/model_pretrained_sub'+str(N_subj_target)
    model.load_state_dict(torch.load(filename))


def training_sub_selection(N_subj_target,corr_mat_sub,top_number):

    pearson_corr_coeff_vec = corr_mat_sub[N_subj_target-1,:]
    print("pearson_corr_coeff_vec : ",pearson_corr_coeff_vec)
    subj_sorted = np.argsort(pearson_corr_coeff_vec)
    print("sub_sorted : ",subj_sorted)
    subj_sorted = subj_sorted[:len(subj_sorted)-1]              # Remove the last element
    print("sub_sorted after removing last : ",subj_sorted)

    N_subj_selec = subj_sorted[len(subj_sorted)-top_number:len(subj_sorted)]

    print("Subject selectionned : ",N_subj_selec)

    return N_subj_selec

def pretrain_TL(N_subj_target,N_subj_selec_arr,root_dir,datatype,t_start,t_end,sample_rate,robust_scaler,chans,samples,n_class,F1,D,F2,bias,dropout,activation,lr,n_gpu,batch_size,epochs,save_dir_pretrained):
    ########################## Pretraining #################################"""" 
    print(" Subject selectioned for subject "+str(N_subj_target) +" : ",N_subj_selec_arr[N_subj_target-1])
    count = 0
    for N_S in N_subj_selec_arr[N_subj_target-1]:
        print("Subject source : ",N_S)
        #Select data path
        data, labels = load_innerspeech_data(root_dir,datatype,N_S,t_start,t_end,sample_rate,False)

        print("size source data :",data.shape)
        print("size source labels :", labels.shape)

        if count==0:
            data_pre = data
            labels_pre = labels
            count = 1
        else:
            data_pre = np.concatenate((data_pre,data),axis = 0)
            labels_pre = np.concatenate((labels_pre,labels))


    #Center and scale all data selected
    if robust_scaler:
        print("Centering and scaling the selected data ...")
        for trials in range(data_pre.shape[0]):
            transformer = RobustScaler().fit(data_pre[trials,:,:])
            data_tr = transformer.transform(data_pre[trials,:,:])
            data_pre[trials,:,:] = data_tr
            data_tr = None
    
    data_pre, labels_pre = shuffle_in_unison(data_pre,labels_pre)

    print("size data_pre : ",data_pre.shape)
    print("size labels_pre : ",labels_pre.shape)

    
    # Train the CNN with all selected data
    # convert data to  (trials, kernels, channels, samples) format. Data 
    # contains 128 channels and 1153 time-points. Set the number of kernels to 1
    X_pretrain      = data_pre.reshape(data_pre.shape[0],kernels, chans, samples)

    print('X_pretrain shape:', X_pretrain.shape)
    print('Y_pretrain shape:', labels_pre.shape)
    print(X_pretrain.shape[0], 'pretrain samples')

    X_pretrain = X_pretrain.astype("float32")
    Y_pretrain = labels_pre.astype("float32").reshape(labels_pre.shape[0],)

    X_pretrain, Y_pretrain = Variable(torch.from_numpy(X_pretrain)),Variable(torch.from_numpy(Y_pretrain))
    Y_pretrain = torch.tensor(Y_pretrain, dtype=torch.long) 

    model_pretrained = EEGNet(n_output = n_class,sample_rate = sample_rate,F1 = F1,D = D,F2 = F2,n_channels = chans,bias = bias,dropout = dropout,activation = activation)
        
    print(model_pretrained)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model_pretrained.parameters(),lr = lr)

    model_pretrained.cuda(n_gpu)
    device = torch.device("cuda:"+str(n_gpu) if torch.cuda.is_available() else "cpu")
    X_pretrain,Y_pretrain = X_pretrain.to(device),Y_pretrain.to(device)

    for epoch in range(epochs):
        model_pretrained.train()

        for i in range(X_pretrain.shape[0]//batch_size):
            start = i*batch_size
            end = (i+1)*batch_size

            data_pretrain = X_pretrain[start:end,:,:]
            labels_pretrain = Y_pretrain[start:end,]

            Y_pred = model_pretrained(data_pretrain)

            loss = criterion(Y_pred, labels_pretrain)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        print("Epoch pretraining : ",epoch+1)
    
    #save_model_pretrained(model_pretrained,save_dir_pretrained,N_subj_target)
    return model_pretrained

def classify_K_fold_TL(N_subj_target,root_dir,datatype,t_start,t_end,sample_rate,robust_scaler,n_class,F1,D,chans,bias,dropout,activation,save_dir_pretrained,corr_mat_sub,top_number,n_splits,samples,transfer_learning,lr,n_gpu,epochs,batch_size,save_dir):
    print("Target subject : ",N_subj_target)
    # Select target subject data and label
    data_target, labels_target = load_innerspeech_data(root_dir,datatype,N_subj_target,t_start,t_end,sample_rate,False)

    model_pretrained = pretrain_TL(N_subj_target,N_subj_selec_arr,root_dir,datatype,t_start,t_end,sample_rate,robust_scaler,chans,samples,n_class,F1,D,F1*D,bias,dropout,activation,lr,n_gpu,batch_size,epochs,save_dir_pretrained)

    # model_pretrained = EEGNet(n_output = n_class,sample_rate = sample_rate,F1=F1,D=D,F2=F1*D,n_channels=chans,bias=bias,dropout=dropout,activation=activation)
    # load_model_pretrained(model_pretrained,save_dir_pretrained,N_subj_target)

    ######## Target training #############

    N_subj_best_corr =training_sub_selection(N_subj_target,corr_mat_sub,top_number)
    print("Subject selectionned for finetunned ",N_subj_best_corr)
    data,labels = data_target, labels_target
    
    for N_S in N_subj_best_corr:
        data_corr, labels_corr = load_innerspeech_data(root_dir,datatype,N_S+1,t_start,t_end,sample_rate,False)
        print("data_corr size : ",data_corr.shape)
        data = np.concatenate((data,data_corr),axis = 0)
        labels = np.concatenate((labels,labels_corr))

    print("data concatenated : ",data.shape)
    
    #Center and scale the data
    if robust_scaler:
        print("Centering and scaling the data ...")
        for trials in range(data.shape[0]):
            transformer = RobustScaler().fit(data[trials,:,:])
            data_tr = transformer.transform(data[trials,:,:])
            data[trials,:,:] = data_tr
            data_tr = None


    data, labels = shuffle_in_unison(data,labels)

    data_fold_arr = np.split(data,n_splits,axis = 0)
    labels_fold_arr = np.split(labels, n_splits, axis = 0)

    for offset in range(1,n_splits+1):
        print("Test fold number ",n_splits-offset+1)

        if offset > 1:
            model_target.reset_parameters(bias)
        
        model_target = model_pretrained
        
        X_test = data_fold_arr[n_splits-offset]
        Y_test = labels_fold_arr[n_splits-offset]
        X_train_arr = [x for i,x in enumerate(data_fold_arr) if i!=n_splits-offset]
        Y_train_arr = [x for i,x in enumerate(labels_fold_arr) if i!=n_splits-offset]

        X_train, Y_train = np.vstack(X_train_arr), np.hstack(Y_train_arr)

        n = X_train.shape[0]

        # print("X_train_shape : ",X_train.shape)
        # print("Y_train_shape :",Y_train.shape)
        # print("X_test_shape : ",X_test.shape)
        # print("Y_test_shape :",Y_test.shape)

        ############################# EEGNet portion ##################################

        # convert data to NHWC (trials, kernels, channels, samples) format. Data 
        # contains 128 channels and 1153 time-points. Set the number of kernels to 1.
        X_train      = X_train.reshape(X_train.shape[0],kernels, chans, samples)
        X_test       = X_test.reshape(X_test.shape[0],kernels, chans, samples)

        print('X_train shape:', X_train.shape)
        print('Y_train shape:', Y_train.shape)
        print(X_train.shape[0], 'train samples')
        print(X_test.shape[0], 'test samples')

        X_train = X_train.astype("float32")
        Y_train = Y_train.astype("float32").reshape(Y_train.shape[0],)

        # train_data.shape = (essai,1,canaux,samples)

        X_train, Y_train = Variable(torch.from_numpy(X_train)),Variable(torch.from_numpy(Y_train))
        Y_train = torch.tensor(Y_train, dtype=torch.long) 

        print(model_target)

        criterion = nn.NLLLoss()

        # Freeze some layers depending on the transfer learning mode
        if transfer_learning == "feature_extraction":
            # Initial layers are frozen (until conv2)
            model_target.conv1.requires_grad = False
            model_target.norm1.requires_grad = False
        elif transfer_learning == "fine_tunning":
            # Final layers are frozen (from conv2)
            model_target.conv2.requires_grad = False
            model_target.norm2.requires_grad = False
            model_target.depthwise.requires_grad = False
            model_target.pointwise.requires_grad = False
            model_target.norm3.requires_grad = False
            if bias:
                model_target.classifier.requires_grad = False


        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_target.parameters()), lr=lr)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,300,500], gamma=0.1)

        model_target.cuda(n_gpu)
        device = torch.device("cuda:"+str(n_gpu) if torch.cuda.is_available() else "cpu")
        X_train,Y_train = X_train.to(device),Y_train.to(device)
        test_loss_history = []
        train_loss_history = []
        train_accuracy_history = []
        test_accuracy_history = []


        for epoch in range(epochs):
            model_target.train()

            epoch_train_loss = 0
            for i in range(X_train.shape[0]//batch_size):
                start  = i*batch_size
                end = (i+1)*batch_size

                data_train = X_train[start:end,:,:]
                labels_train = Y_train[start:end,]

                Y_pred = model_target(data_train)

                loss = criterion(Y_pred, labels_train)
                epoch_train_loss += loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            model_target.eval()

            # Training evaluation
            train_loss = epoch_train_loss.item()/(X_train.shape[0]//batch_size)
            train_loss_history.append(train_loss)

            n = labels_train.shape[0]
            correct = (torch.max(Y_pred,1)[1]==labels_train).sum().item()
            train_accuracy = correct / n
            train_accuracy_history.append(train_accuracy)

            
            # Testing evaluation
            test_accuracy,test_loss = testing(X_test,Y_test,model_target,device,criterion,n_gpu)
            test_accuracy_history.append(test_accuracy)
            test_loss_history.append(test_loss)

            print("epochs:",epoch+1,"Test Loss",test_loss,"Train Loss",train_loss,"Training Accuracy:",train_accuracy,"Testing Accuracy:",test_accuracy,"Learning rate:",scheduler.get_last_lr()[0])

        df = pd.DataFrame({"train_loss_history":train_loss_history,"test_loss_history":test_loss_history,"train_accuracy_history":train_accuracy_history,"test_accuracy_history":test_accuracy_history})

        file_path_csv = save_dir + '/Scores_sub_'+str(N_subj_target)+'_fold_'+str(n_splits-offset+1)+'.csv'
        df.to_csv(file_path_csv,encoding="utf-8-sig")

        save_model(model_target,save_dir,N_subj_target,n_splits-offset+1)


#Parameters
n_class = 4
sample_rate = 256
F1 = 32
D = 16
F2 = D*F1
lr = 0.001
epochs = 50
n_splits = 4
robust_scaler = True
n_gpu = 0
datatype = "EEG"
t_start = 0
t_end = 4.5
kernels, chans, samples = 1, 128, int((t_end - t_start)*sample_rate)
batch_size = 10
bias = True
dropout = 0.25
activation = "ELU"

transfer_learning_arr = ["fine_tunning","feature_extraction"] # Just 2 modes feature_extraction or fine_tunning
top_number_arr = [1,2,3,4,5]

# Subjects
N_subj_target_arr = [1,2]

# Selected subjects for each subject target computed with Corr.py (line 1 = subj 1 ...)
N_subj_selec_arr = np.array([[2,3,4,5,9],[1,3,5,8,9],[1, 4, 6, 8, 9],[1, 3, 5, 6, 9],[1, 2, 4, 6, 9],[3, 4, 5, 9],[4, 6, 10],[1, 2, 3, 4, 9],[1, 3, 4, 5, 6],[1, 3, 4, 6, 7, 9]])

# Correlation matrix for all the subjects
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


# Path of Preprocessed data
root_dir = '../Inner_Speech_Database'
save_dir_pretrained = './model_pretrained'


for N_subj_target in N_subj_target_arr:
    for transfer_learning in transfer_learning_arr:
        for top_number in top_number_arr:
            print("transfer_learning : ",transfer_learning)
            print("top : ",top_number)
            save_dir = transfer_learning +'_'+'top'+str(top_number)+'_' +str(activation)+'_dropout='+str(dropout)+'_F1='+str(F1)+'_D='+str(D)+'_lr='+str(lr)+'_bs='+str(batch_size)+'_bias='+str(bias)+'_'+str(chans)+'chansv2'
        
            classify_K_fold_TL(N_subj_target,root_dir,datatype,t_start,t_end,sample_rate,robust_scaler,n_class,F1,D,chans,bias,dropout,activation,save_dir_pretrained,corr_mat_sub,top_number,n_splits,samples,transfer_learning,lr,n_gpu,epochs,batch_size,save_dir)
