import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#Parameters
chans = 128       # 128 /26 channels
n_splits = 4
N_subj_arr = [1,2,3,4,5,6,7,8,9,10]

activation_arr = ["ELU","SILU","GELU"]
batch_size_arr = [4,10,20]

F1_D_arr = [[16,4],[24,8],[32,16]]
lr_arr = [0.1, 0.01, 0.001, 0.0001]
dropout_arr = [0.25,0.5]
bias_arr = [False,True]

n_tot_exp = len(activation_arr)*len(batch_size_arr)*len(lr_arr)*len(dropout_arr)*len(bias_arr)*len(F1_D_arr)
count_exp = 0
acc_max = 0
dir = 'C:/Users/guill/Documents/ENSEIRB/Stage_2A/InnerSpeech/Classification/Scores/Inner/All_exp/'
acc_arr = []
path_arr = []
best_path_arr = []
acc_max_arr = []

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

                        path = activation+'_dropout='+str(dropout)+'_F1='+str(F1)+'_D='+str(D)+'_lr='+str(lr)+'_bs='+str(batch_size)+'_bias='+str(bias)+'_'+str(chans)+'chans'+'_NLLLoss'+'/'
                        path_arr.append(path)

                        acc_tot = 0
                        std_tot = 0
                        for N_S in N_subj_arr:
                            EEGNet_ELU1 = pd.DataFrame(pd.read_csv(dir+path+'Scores_sub_'+str(N_S)+'_fold_1.csv',encoding="utf-8-sig"))
                            EEGNet_ELU2 = pd.DataFrame(pd.read_csv(dir+path+'Scores_sub_'+str(N_S)+'_fold_2.csv',encoding="utf-8-sig"))
                            EEGNet_ELU3 = pd.DataFrame(pd.read_csv(dir+path+'Scores_sub_'+str(N_S)+'_fold_3.csv',encoding="utf-8-sig"))
                            EEGNet_ELU4 = pd.DataFrame(pd.read_csv(dir+path+'Scores_sub_'+str(N_S)+'_fold_4.csv',encoding="utf-8-sig"))

                            avr1,std1 = np.mean(np.array(EEGNet_ELU1['test_accuracy_history'])*100),np.std(np.array(EEGNet_ELU1['test_accuracy_history'])*100)
                            avr2,std2 = np.mean(np.array(EEGNet_ELU2['test_accuracy_history'])*100),np.std(np.array(EEGNet_ELU2['test_accuracy_history'])*100)
                            avr3,std3 = np.mean(np.array(EEGNet_ELU3['test_accuracy_history'])*100),np.std(np.array(EEGNet_ELU3['test_accuracy_history'])*100)
                            avr4,std4 = np.mean(np.array(EEGNet_ELU4['test_accuracy_history'])*100),np.std(np.array(EEGNet_ELU4['test_accuracy_history'])*100)

                            acc = (avr1 + avr2 + avr3 + avr4)/n_splits
                            std = (std1 + std2 + std3 + std4)/n_splits
                            #print("Testing Accuracy subject "+str(N_S)+" :"+str(acc)+" ± "+str(std))
                            acc_tot = acc_tot + acc
                            std_tot = std_tot + std

                        acc_tot = acc_tot/len(N_subj_arr)
                        std_tot = std_tot/len(N_subj_arr)
                        acc_arr.append(acc_tot)
                        
                        #print("Testing Accuracy all : "+str(acc_tot)+" ± "+str(std_tot))
                        if acc_tot > acc_max:
                            acc_max = acc_tot
                            best_path = path
                            # best_path_arr.append(best_path)
                            # acc_max_arr.append(acc_max)

# Print best model
print("Best parameters : "+ best_path)
acc_tot = 0
std_tot = 0
for N_S in N_subj_arr:
    EEGNet_ELU1 = pd.DataFrame(pd.read_csv(dir+best_path+'Scores_sub_'+str(N_S)+'_fold_1.csv',encoding="utf-8-sig"))
    EEGNet_ELU2 = pd.DataFrame(pd.read_csv(dir+best_path+'Scores_sub_'+str(N_S)+'_fold_2.csv',encoding="utf-8-sig"))
    EEGNet_ELU3 = pd.DataFrame(pd.read_csv(dir+best_path+'Scores_sub_'+str(N_S)+'_fold_3.csv',encoding="utf-8-sig"))
    EEGNet_ELU4 = pd.DataFrame(pd.read_csv(dir+best_path+'Scores_sub_'+str(N_S)+'_fold_4.csv',encoding="utf-8-sig"))

    avr1,std1 = np.mean(np.array(EEGNet_ELU1['test_accuracy_history'])*100),np.std(np.array(EEGNet_ELU1['test_accuracy_history'])*100)
    avr2,std2 = np.mean(np.array(EEGNet_ELU2['test_accuracy_history'])*100),np.std(np.array(EEGNet_ELU2['test_accuracy_history'])*100)
    avr3,std3 = np.mean(np.array(EEGNet_ELU3['test_accuracy_history'])*100),np.std(np.array(EEGNet_ELU3['test_accuracy_history'])*100)
    avr4,std4 = np.mean(np.array(EEGNet_ELU4['test_accuracy_history'])*100),np.std(np.array(EEGNet_ELU4['test_accuracy_history'])*100)

    acc = (avr1 + avr2 + avr3 + avr4)/n_splits
    std = (std1 + std2 + std3 + std4)/n_splits
    print("Testing Accuracy subject "+str(N_S)+" :"+str(acc)+" ± "+str(std))
    acc_tot = acc_tot + acc
    std_tot = std_tot + std

acc_tot = acc_tot/len(N_subj_arr)
std_tot = std_tot/len(N_subj_arr)
                        
print("Testing Accuracy all : "+str(acc_tot)+" ± "+str(std_tot))
print("\n")

acc_np = np.array(acc_arr)
index_arr = np.argsort(acc_np)

rank = len(index_arr)
for index in index_arr:
    print(str(rank)+") "+path_arr[index] + "with an acc of "+str(acc_arr[index]))
    rank-=1


    # plt.plot(EEGNet_ELU['loss'], '-c', label='EEGNet')
    # plt.xlabel("Epoch",fontsize=13)
    # plt.legend(loc='best')

    # plt.ylabel("Loss Value",fontsize=13)
    # plt.title("Loss Curve",fontsize=18)
    # plt.savefig('Loss'+'_sub'+str(N_S)+'.jpg')

    # plt.show()
    # plt.close()

    # plt.plot(np.array(EEGNet_ELU['train_accuracy_history'])*100, '-b', label='train')
    # plt.plot(np.array(EEGNet_ELU['test_accuracy_history'])*100, '-c', label='test')

    # plt.xlabel("Epoch",fontsize=13)
    # plt.legend(loc='best')

    # plt.ylabel("Accuracy(%)",fontsize=13)
    # plt.title('EEGNet Performance for sub'+str(N_S),fontsize=18)
    # plt.savefig('Accuracy'+'_sub'+str(N_S)+'.jpg')

    # plt.show()
    # plt.close()