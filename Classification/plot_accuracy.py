import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#from Classify_K_fold_1sub import save_model


#Parameters
chans = 128       # 128 /26 channels
n_splits = 4
N_subj_arr = [1,2,3,4,5,6,7,8,9,10]

activation = "ELU"
batch_size = 10
F1 = 32
D = 16
lr = 0.001
dropout = 0.25
bias = True
transfer_learning = "fine_tunning"
top_number_arr = [1,2,3,4,5,6,7,8,9]
save_dir = 'C:/Users/guill/Documents/ENSEIRB/Stage_2A/InnerSpeech/Classification/Scores/Inner/'

for top_number in top_number_arr:

    print("transfer_learning:",transfer_learning,"top:",top_number,"activation:",activation,"batch_size:",batch_size,"lr:",lr,"dropout:",dropout,"bias:",bias,"F1:",F1,"D:",D,"chans:",chans)
    print("\n")

    #path = save_dir +activation+'_dropout='+str(dropout)+'_F1='+str(F1)+'_D='+str(D)+'_lr='+str(lr)+'_bs='+str(batch_size)+'_bias='+str(bias)+'_'+str(chans)+'chans'+'_NLLLoss'+'/'
    path = save_dir+transfer_learning +'_'+'top'+str(top_number)+'_' +str(activation)+'_dropout='+str(dropout)+'_F1='+str(F1)+'_D='+str(D)+'_lr='+str(lr)+'_bs='+str(batch_size)+'_bias='+str(bias)+'_'+str(chans)+'chans/'

    acc_tot = 0
    std_tot = 0
    for N_S in N_subj_arr:
        EEGNet_ELU1 = pd.DataFrame(pd.read_csv(path+'Scores_sub_'+str(N_S)+'_fold_1.csv',encoding="utf-8-sig"))
        EEGNet_ELU2 = pd.DataFrame(pd.read_csv(path+'Scores_sub_'+str(N_S)+'_fold_2.csv',encoding="utf-8-sig"))
        EEGNet_ELU3 = pd.DataFrame(pd.read_csv(path+'Scores_sub_'+str(N_S)+'_fold_3.csv',encoding="utf-8-sig"))
        EEGNet_ELU4 = pd.DataFrame(pd.read_csv(path+'Scores_sub_'+str(N_S)+'_fold_4.csv',encoding="utf-8-sig"))

        avr1,std1 = np.mean(np.array(EEGNet_ELU1['test_accuracy_history'])*100),np.std(np.array(EEGNet_ELU1['test_accuracy_history'])*100)
        avr2,std2 = np.mean(np.array(EEGNet_ELU2['test_accuracy_history'])*100),np.std(np.array(EEGNet_ELU2['test_accuracy_history'])*100)
        avr3,std3 = np.mean(np.array(EEGNet_ELU3['test_accuracy_history'])*100),np.std(np.array(EEGNet_ELU3['test_accuracy_history'])*100)
        avr4,std4 = np.mean(np.array(EEGNet_ELU4['test_accuracy_history'])*100),np.std(np.array(EEGNet_ELU4['test_accuracy_history'])*100)

        acc = (avr1 + avr2 + avr3 + avr4)/n_splits
        std = (std1 + std2 + std3 + std4)/n_splits
        print("Testing Accuracy subject "+str(N_S)+" :"+str(acc)+" ± "+str(std))
        # print("Accuracy fold 1 :"+str(avr1))
        # print("Accuracy fold 2 :"+str(avr2))
        # print("Accuracy fold 3 :"+str(avr3))
        # print("Accuracy fold 4 :"+str(avr4))
        acc_tot = acc_tot + acc
        std_tot = std_tot + std

    acc_tot = acc_tot/len(N_subj_arr)
    std_tot = std_tot/len(N_subj_arr)
                                
    print("Testing Accuracy all : "+str(acc_tot)+" ± "+str(std_tot))

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