import HazardNet_Utils
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
import datetime
from sklearn import metrics
import importlib
importlib.reload(HazardNet_Utils)

##########################
#  Model Size            #
##########################
class ModelSize():
    def __init__(self, mdl):
        self.model = mdl
        self.size_1 = self.get_size_1()
        self.size_2 = self.get_size_2()

    def get_size_1(self):
        w=[]; n=[]; wn=[]
        model_named_parameters = self.model.named_parameters()
        # print('\n\n _0_ Model Size')
        for name , weight in model_named_parameters:
            # print('_1_',name,weight.data.cpu().numpy().shape)
            wgh = np.prod(weight.data.cpu().numpy().shape)
            # print('_2_',wgh)
            # print(wgh, name, weight.data.cpu().numpy().shape)
            wn = np.append(wn, [str(name), str(wgh), str(weight.data.cpu().numpy().shape)])
            # print('_3_',wn)
            w = np.append(w, wgh)
            # print('_4_',w)
        # print('Model Size 1: ', w.sum())
        return w.sum(), w, wn


    def get_size_2(self):
        size = 0
        for param in self.model.parameters():
            size += param.data.numpy().size
        # print('Model Size 2: ', size) 
        return size

        # 
           


##########################
#  Metrics               #
##########################

def metrics_all_msk(TN, FN, FP, TP):
    '''
    sum_dt, acc, precision, recall, f1, MCC = metrics_all_msk(TN, FN, FP, TP)
    '''
    
    TN = float(TN); FN= float(FN); FP= float(FP); TP= float(TP)
    sum_dt = TN + FN + FP + TP
    acc = 100*(TN+TP)/sum_dt
    try: 
        precision = TP/(TP+FP)
    except: 
        precision = np.nan  
    try: 
        recall =  TP/(TP+FN)
    except: 
        recall = np.nan
    try: 
        f1 = 2*(precision*recall)/(precision+recall)
    except: 
        f1 = np.nan   
    try: 
        MCC = ((TP*TN )- (FP*FN))/(np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
    except: 
        MCC = np.nan
    return sum_dt,acc,precision,recall,f1,MCC 
 
        
def conf_mat(y, y_pred):
    '''
    TN, FP, FN, TP = conf_mat(y, y_pred)
    '''
    confusion_matrix = metrics.confusion_matrix(y, y_pred)
    if confusion_matrix.size == 1:
        if not np.sum(y):
            TN = confusion_matrix[0][0]; FP = 0; FN = 0; TP = 0
        if np.sum(y):
            TN = 0; FP = 0; FN = 0; TP = confusion_matrix[0][0]
    else:
        TN = confusion_matrix[0][0]
        FP = confusion_matrix[0][1]
        FN = confusion_matrix[1][0]
        TP = confusion_matrix[1][1]
    return TN, FP, FN, TP




    

##########################
#  1DXC data - 1DConv    #
##########################

class thermal_dataset_1d(Dataset):
    '''
    Class for 1DConv
    '''

    def __init__(self, X_data, y_data, TW_data_num):
        self.X_data = X_data
        self.y_data = y_data
        self.TW_data_num = TW_data_num

    def oneSample(self, index):
        index = index + self.TW_data_num
        input_data = self.X_data[index-self.TW_data_num:index]
        input_data = np.swapaxes(input_data, 0, 1)

        label_data = self.y_data[index]
        # input_data = np.expand_dims(input_data, axis=0)
        # input_data = np.squeeze(input_data, axis = 0)
        # print('input_data.shape',input_data.shape)
        return input_data, label_data


    def __getitem__(self, index):
        return self.oneSample(index)

    def __len__ (self):
        return len(self.X_data)-self.TW_data_num


##########################
#  2DXC data - 2DConv    #
##########################
class thermal_dataset_2d(Dataset):
    """
    Class for 2DConv
    """

    def __init__(self, X_data, y_data, TW_data_num):
        self.X_data = X_data
        self.y_data = y_data
        self.TW_data_num = TW_data_num

    def oneSample(self, index):
        index = index + self.TW_data_num
        input_data = self.X_data[index-self.TW_data_num:index]
        label_data = self.y_data[index]
        input_data = np.expand_dims(input_data, axis=0)
        # print(f'input_data.shape {input_data.shape} label_data {label_data}')
        return input_data, label_data
    


    def __getitem__(self, index):
        return self.oneSample(index)

    def __len__ (self):
        return len(self.X_data)-self.TW_data_num
    

##########################
#    3D data - 3DConv    #
##########################

class thermal_dataset_3d(Dataset):
    '''
    '''
    def __init__(self, data_4d, y_data, TW_data_num):
        self.data_4d = data_4d
        self.y_data = y_data
        self.TW_data_num = TW_data_num

    def oneSample(self, index):
        index = index + self.TW_data_num
        input_data = np.array([self.data_4d[index-self.TW_data_num:index]])
        label_data = self.y_data[index]
#         input_data = np.expand_dims(input_data, axis=0)
        input_data = np.squeeze(input_data, axis = 0)
        return input_data, label_data

    def __getitem__(self, index):
        return self.oneSample(index)

    def __len__ (self):
        return len(self.data_4d)-self.TW_data_num
    

# create dataset
def train_dataloader(X,y,TW_data_num,replacement,batch_size,sampler_active,print_txt_file_name, Conv,num_workers=16):
    if Conv=='1D':
        df = thermal_dataset_1d(X, y, TW_data_num)
    elif Conv=='2D':
        df = thermal_dataset_2d(X, y, TW_data_num)
    elif Conv=='3D':
        df = thermal_dataset_3d(X, y, TW_data_num)
    else:
        print('Conv should be 1D or 2D or 3D')
    new_y = y[TW_data_num:]
    # frist data are canceled due to that we could not create input temp data for first TW=360 rows 
    # class_sample_count counts the number of samples per class 1 and 0
    class_sample_count = np.array([len(np.where(new_y == t)[0]) for t in np.unique(new_y)])
    HazardNet_Utils.prnt_w2f(f'class_sample_count: {class_sample_count}',print_txt_file_name)

    weight = 1. / class_sample_count
    HazardNet_Utils.prnt_w2f(f'weight{weight}',print_txt_file_name)



    samples_weight = np.array([weight[t] for t in new_y])
    HazardNet_Utils.prnt_w2f(f'len(samples_weight): {len(samples_weight)}',print_txt_file_name)





    samples_weight = torch.from_numpy(samples_weight).double()
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight,
                                                     int(len(samples_weight)),
                                                     replacement=replacement)
    
    if sampler_active:
        data_loader_train = DataLoader(dataset=df,
                                       batch_size=batch_size,
                                       shuffle=False, 
                                       sampler=sampler,
                                       num_workers=num_workers)
        HazardNet_Utils.prnt_w2f('sampler_active',file_name=print_txt_file_name)
    else:
        data_loader_train = DataLoader(dataset=df,
                                       batch_size=batch_size,
                                       shuffle=True, # TODO 
                                       num_workers=num_workers)
        HazardNet_Utils.prnt_w2f('sampler_deactive',file_name=print_txt_file_name)




    return data_loader_train



def val_dataloader(X,y,TW_data_num,batch_size, Conv, num_workers=16):
    '''
    '''
    if Conv=='1D':
        df = thermal_dataset_1d(X, y, TW_data_num)
    elif Conv=='2D':
        df = thermal_dataset_2d(X, y, TW_data_num)
    elif Conv=='3D':
        df = thermal_dataset_3d(X, y, TW_data_num)
    else:
        print('Conv should be 1D or 2D or 3D')

    data_loader_val = DataLoader(dataset=df,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers)
    return data_loader_val 







##########################
#  Train and Validate    #
##########################

def train_batch(model, criterion, data_loader_train, device, optimizer):
    TN_train=0; FN_train=0; FP_train=0; TP_train=0; train_epoch_loss=0; balanced_weight_training = 0

    for X_train_batch, y_train_batch in data_loader_train:
#       NOT very accurate !!! Related to the batch size !!! 
        balanced_weight_training += y_train_batch.data.cpu().numpy().sum()*100/(len(y_train_batch)*len(data_loader_train))
        
    #   X_train_batch = X_train_batch.transpose(1,2) cancell due to the 2DConv
#         print('X_train_batch',X_train_batch.shape, 'y_train_batch',y_train_batch.shape)
        
        X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
        optimizer.zero_grad()
        y_train_pred = model(X_train_batch)
        train_loss = criterion(y_train_pred, y_train_batch)
        train_loss.backward()
        optimizer.step()
            # confusion matrix and log
        _, y_pred_tags_train = torch.max(torch.log_softmax(y_train_pred, dim = 1), dim = 1)
        TN, FP, FN, TP = conf_mat(y_train_batch.data.cpu().numpy(), y_pred_tags_train.data.cpu().numpy())
        TN_train += TN
        FP_train += FP
        FN_train += FN
        TP_train += TP
        train_epoch_loss += train_loss.item()/len(data_loader_train)
    return  TN_train, FN_train, FP_train, TP_train, train_epoch_loss, balanced_weight_training


def validation_batch(model, data_loader_val, device, criterion):
    TN_val=0; FN_val=0; FP_val=0; TP_val=0; val_epoch_loss=0
    for X_val_batch, y_val_batch in data_loader_val:
#       X_val_batch = X_val_batch.transpose(1,2) cancell due to the 2DConv
        X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
        y_val_pred = model(X_val_batch)
        val_loss = criterion(y_val_pred, y_val_batch)
        # confusion matrix and log
        _, y_pred_tags_val = torch.max(torch.log_softmax(y_val_pred, dim = 1), dim = 1)
        TN, FP, FN, TP = conf_mat(y_val_batch.data.cpu().numpy(), y_pred_tags_val.data.cpu().numpy())
        TN_val += TN; FP_val += FP; FN_val += FN; TP_val += TP
        val_epoch_loss += val_loss.item()/len(data_loader_val)
    return TN_val, FN_val, FP_val, TP_val, val_epoch_loss

def train_msk(epochs, patience_early_stop, model,criterion, data_loader_train, device, optimizer, data_loader_val, result_path, optimizer_scheduler, print_txt_file_name, save_results=False):
    # patience_early_stop: Number of epochs to wait for improvement
    best_val_loss = float('inf')
    early_stop_counter = 0  # Counter to track how many epochs without improvement


    log_index=0
    confMatrix_log = pd.DataFrame(columns=['epoch','TN_train','FP_train','FN_train','TP_train','loss_train','TN_val','FP_val','FN_val','TP_val','loss_val'])
    start_time = datetime.datetime.now()
    for epoch in range(1, epochs+1):
        # TRAIN THE MODEL
        model.train()
        TN_train, FN_train, FP_train, TP_train, train_epoch_loss, balanced_weight_training = train_batch(model, criterion, data_loader_train, device, optimizer)
        optimizer_scheduler.step()
        # Get the current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        HazardNet_Utils.prnt_w2f(f'In epoch {epoch} the learning rate of optimizer is {current_lr}.',file_name=print_txt_file_name)
#         print(f'In epoch {epoch} the learning rate of optimizer is{get_lr(optimizer)}.')


        with torch.no_grad():
            model.eval()
            # METRICS FOR VALIDATION
            
            TN_val=0; FN_val=0; FP_val=0; TP_val=0; val_epoch_loss=0
            TN_val, FN_val, FP_val, TP_val, val_epoch_loss = validation_batch(model, data_loader_val, device, criterion)
            
            _, acc_val, _, _, _, _ = metrics_all_msk(TN=TN_val, FN=FN_val, FP=FP_val, TP=TP_val)
            _, acc_train, _, _, _, _ = metrics_all_msk(TN=TN_train, FN=FN_train, FP=FP_train, TP=TP_train)
            
            
            HazardNet_Utils.prnt_w2f(f'Epoch {epoch+0:03}: | Acc_tr: {acc_train:.2f} | Loss_tr: {train_epoch_loss:.2f}\
            | Acc_val: {acc_val:.2f} | Loss_val: {val_epoch_loss:.2f}',file_name=print_txt_file_name)
        
        confMatrix_log.loc[log_index,['epoch',
                                      'TN_train','FN_train','FP_train','TP_train','loss_train',
                                      'TN_val','FN_val','FP_val','TP_val','loss_val']] =\
        [epoch,
         TN_train, FN_train, FP_train, TP_train, train_epoch_loss, 
         TN_val,   FN_val,   FP_val,   TP_val,   val_epoch_loss]
        
        log_index+=1

        if save_results:
            if epoch%10==0: 
                confMatrix_log.to_csv(result_path+'/confMatrix_log.csv')
                torch.save(model, result_path+'/model_epoch_'+str(epoch)+'.pt')
                epoch_time = datetime.datetime.now()
                HazardNet_Utils.prnt_w2f(f'Epoch {epoch} - Time Delta: {epoch_time-start_time}',file_name=print_txt_file_name)
                f= open(result_path+'/log.txt','a+')
                f.write('\nepoch'+str(epoch)+' Start time: '+str(start_time)+' Epoch time: '+str(epoch_time)+' Training Time: '+str(epoch_time-start_time))
                f.close()

        # Early Stopping & Save Best Model 
        # Check if validation loss has improved
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            early_stop_counter = 0
            # Save the best model checkpoint if needed
            # TODO: I should update to save just last best model 'best_model'+str(epoch)+'.pt' ==> 'best_model.pt'
            torch.save(model, result_path+'/best_model'+str(epoch)+'.pt')
        else:
            early_stop_counter += 1

        # Check if early stopping criteria are met
        if early_stop_counter >= patience_early_stop:
            HazardNet_Utils.prnt_w2f(f'Early stopping at epoch {epoch}',file_name=print_txt_file_name)
            break
