import sys
sys.path.append("./src/")
import numpy as np
from torchsummary import summary
import torch
import torch.nn as nn
import torch.optim as optim
import importlib
import HazardNet_Models
import HazardNet_DataGen
import HazardNet_Utils
import HazardNet_DL_Helpers
import random
    

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def train_model(result_path, 
                patience_early_stop, 
                device, 
                lr, 
                dropout, 
                X_train, 
                lbe_train, 
                LAMBDA, 
                optimizer_step_size, 
                gamma, 
                batch_size_train, 
                batch_size_val, 
                epochs, 
                replacement,
                train_val_period, 
                print_txt_file_name):

    HazardNet_Utils.prnt_w2f(f'device: {device}', print_txt_file_name)
    # Train and validation data
    # Here it is important to set sampler_active=False or True based on the decsion.
    train_dt =  HazardNet_DL_Helpers.train_dataloader(X=X_train, 
                                                      y=lbe_train, 
                                                      TW_data_num=36, 
                                                      batch_size=batch_size_train, 
                                                      replacement=replacement, 
                                                      num_workers=16, 
                                                      sampler_active=True,
                                                      print_txt_file_name=print_txt_file_name, 
                                                      Conv='3D')
    # train_dt =  HazardNet_DL_Helpers.val_dataloader(X=X_train, y=lbe_train, TW_data_num=36, batch_size=batch_size_train, shuffle=True, num_workers=16)
    val_dt =  HazardNet_DL_Helpers.val_dataloader(X=X_val,
                                                  y=lbe_val, 
                                                  TW_data_num=36, 
                                                  batch_size=batch_size_val, 
                                                  num_workers=16, 
                                                  Conv='3D')
    HazardNet_Utils.prnt_w2f(f'Train dataset class 1 weight :{lbe_train.mean()} Val dataset class 1 weight:{lbe_val.mean()}',print_txt_file_name)

    
  
    
    model = HazardNet_Models.Net_3D36C(dropout=dropout).double()  
    
    mdl_sz = HazardNet_DL_Helpers.ModelSize(model)
    (size_of_model_1 , a,b) = mdl_sz.get_size_1()
    
    HazardNet_Utils.prnt_w2f(f'\nSize_of_Model: {size_of_model_1}\na: {a}\nb: {b}\n',print_txt_file_name)
    HazardNet_Utils.prnt_w2f(f'\nmodel: \n\n{summary(model, input_size=(36, 17, 36, 18))}\n',print_txt_file_name)
    HazardNet_Utils.prnt_w2f(f'\nmodel: \n\n{model}\n',print_txt_file_name)

   
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    # Weghited loss function
    # criterion = nn.CrossEntropyLoss(weight=torch.tensor([1,weight],dtype=torch.double).to(device))
    Sigmoid = nn.Sigmoid()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=LAMBDA)
    optimizer_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=optimizer_step_size, gamma=gamma)
    
    
    
    HazardNet_Utils.prnt_w2f(f'\nStart Train Data: {str(train_val_period[0])} \
                             \nStop Train Data: {str(train_val_period[1])} \
                             \nStart Val Data: {str(train_val_period[2])} \
                             \nStop Val Data: {str(train_val_period[3])} \
                             \nBatch size train: {str(batch_size_train)} \
                             \nBatch size validation: {str(batch_size_val)} \
                             \nEpochs: {str(epochs)} \
                             \nOptimizer step size: {str(optimizer_step_size)} \
                             \ngamma: {str(gamma)} \
                             \nLAMBDA: {str(LAMBDA)} \
                             \nDropout: {str(dropout)} \
                             \nLearning Rate: {str(lr)} \
                             \nreadme_text: {str(readme_text)} \
                             \n',print_txt_file_name)


    HazardNet_DL_Helpers.train_msk(epochs = epochs,
                                   patience_early_stop=patience_early_stop, 
                                   model = model,
                                   criterion = criterion,
                                   data_loader_train = train_dt,
                                   data_loader_val=val_dt,
                                   device = device, 
                                   optimizer = optimizer,
                                   optimizer_scheduler =optimizer_scheduler,
                                   result_path=result_path,
                                   save_results=True,
                                   print_txt_file_name=print_txt_file_name)




##################################################################################################
# Parameters

root_path = './TRAIN_RESULTS/Inlet/3D36C_4D_Inlet/'
res_folder_name = 'RESULTS'

readme_text = """
"""
train_val_period = ['2019-05-01','2019-06-01','2019-06-01','2019-06-08']


dvc = 'cuda:3'
STIT = 0.05
lr = 0.01
dropout = 0.01
LAMBDA = 0.001 #L2 regularization
optimizer_step_size = 10
gamma = 0.9 # Learning rate decay after each step_size epoch lr = lr*gamma
batch_size_train = 200
batch_size_val = 200
epochs = 100
replacement = True
patience_early_stop = epochs  # Number of epochs to wait for improvement. Bigger than 10 to save the metrics csv file. When I put epochs it means no early stop. 

# Create path for results and text file for logging all the print commands
result_path, prnt_txt_file = HazardNet_Utils.create_results_dir_txt_file(root_path, res_folder_name, readme_text)


# Traing data
FourD_data = HazardNet_DataGen.FourD_DataGen(months=['05'],
                                             STIT=STIT, 
                                             model_type=['prediction','4D_Inlet'], 
                                             compute_server='lab_gpu', 
                                             data_path=None
                                            )


# Test data
FourD_data_test = HazardNet_DataGen.FourD_DataGen(months=['06'], 
                                                  STIT=STIT,
                                                  model_type=['prediction','4D_Inlet'], 
                                                  compute_server='lab_gpu', 
                                                  data_path=None
                                                  )



X_train, lbe_train = FourD_data.load_data()

#################
#    SMOTE      #   
#################

# Important check here if you want to upsampling with SMOTE
# 
X_train, lbe_train = HazardNet_DataGen.Upsampling_SMOTE(X=X_train, y=lbe_train)
HazardNet_Utils.prnt_w2f(f'After Upsampling SMOTE X_train.shape: {X_train.shape} --- lbe_train.shape: {lbe_train.shape}', prnt_txt_file)
# X_val, lbe_val = FourD_data.first_week_june_data_val()
X_test, lbe_test = FourD_data_test.load_data()
X_val, lbe_val = X_test[:1009], lbe_test[:1009]



HazardNet_Utils.prnt_w2f(f'X_train.shape: {X_train.shape} lbe_train.shape: {lbe_train.shape}\nX_val.shape: {X_val.shape} lbe_val.shape: {lbe_val.shape} \
    \nSTIT: {str(STIT)}',
    prnt_txt_file) 


##################################################################################################  	

device = torch.device(dvc if torch.cuda.is_available() else 'cpu')

train_model(result_path=result_path, 
            patience_early_stop=patience_early_stop, 
            device=device, 
            lr=lr, 
            dropout=dropout, 
            X_train=X_train, 
            lbe_train=lbe_train, 
            LAMBDA=LAMBDA, 
            optimizer_step_size=optimizer_step_size, 
            gamma=gamma, 
            batch_size_train=batch_size_train, 
            batch_size_val=batch_size_val, 
            epochs=epochs, 
            replacement=replacement,
            train_val_period=train_val_period,
            print_txt_file_name=prnt_txt_file)


