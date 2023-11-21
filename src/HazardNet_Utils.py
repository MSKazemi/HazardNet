import sys
sys.path.append("/home/seyedkazemi/codes/HPCRoomModel/mskhelper/")
import datetime as dt
import datetime
from matplotlib import pyplot as plt
import os
import pandas as pd
import numpy as np




##################
# Name Generator ###############################################################
##################

def dir_name_creator(dir_name):
    counter = 0
    while os.path.isdir(dir_name):
        counter += 1
        dir_name = dir_name.split('(')[0]+'('+str(counter)+')'
    return dir_name

def dir_creator(dir_name,readme_text=None):
    dir_name = dir_name_creator(dir_name)
    os.mkdir(dir_name)
    if not readme_text==None:
        readme(dir_name+'/'+dir_name.split('/')[-1],readme_text)#
    print(dir_name,' Created!')
    return dir_name


def readme(readme_name,readme_text):
    f= open(readme_name+".txt","w+")
    f.write(readme_text)
    f.close()
    

def image_name_creator(image_name):
    '''
    Create the image name.
    '''
    counter = 0
    while os.path.isfile(image_name):
        counter += 1
        image_name, image_fromat = image_name.split('.')[0], image_name.split('.')[1]
        
        print(image_name,image_fromat)
        image_name = image_name.split('(')[0]+'('+str(counter)+').'+image_fromat
    print('Image name is :', image_name)
    return image_name


def save_image(image_name='image.jpg'):
    '''
    please use before the plt.show()
    save_image(image_name='image.jpg')
    '''
    image_name = image_name_creator(image_name)
    plt.savefig(image_name,bbox_inches = 'tight', pad_inches = 0.2, dpi=200)
    
    
    
    
def csv_file_name_creator(path, file_name, log=False):
    '''
    Create a file name.
    '''
    counter = 0
    while os.path.isfile(path+file_name):
        counter += 1
        file_name, file_fromat = file_name.split('.csv')[0], file_name.split('.csv')[1:]
        file_fromat = 'csv' + '.'.join(file_fromat)
        file_name = file_name.split('(')[0]+'('+str(counter)+').'+file_fromat
    print('File name is : '+str(file_name))    
    if log == True:
        logging.info('File name is : '+str( file_name))
    return file_name


def file_name_creator(path, file_name, log=False):
    '''
    Create a file name.
    '''
    counter = 0
    while os.path.isfile(path+file_name):
        counter += 1
        file_name, file_fromat = file_name.split('.')[0], file_name.split('.')[1]
        file_name = file_name.split('(')[0]+'('+str(counter)+').'+file_fromat
    print('File name is : '+str(file_name))    
    if log == True:
        logging.info('File name is : '+str(file_name))
    return file_name



def create_results_dir_txt_file(root_path, res_folder_name, readme_text=None):
    # There are three level of folders: root_path, result_path, res_folder_name 

    # Create main folder for results
    if not os.path.isdir(root_path):
        os.makedirs(root_path)


    # Create subfolder for results
    result_path = dir_creator(root_path+res_folder_name,readme_text)
    
    # Create text file for logging all the print commands
    prnt_txt_file = result_path+'/'+res_folder_name+".txt"

    return result_path, prnt_txt_file

##############################
#   Print and Write to file  ######################################################################
##############################

import sys, os

def prnt_w2f(text, file_name, time_stamp=True):
    # print and write to a file
    # Redirect stdout to a file
    original_stdout = sys.stdout
    with open(file_name, 'a' if os.path.exists(file_name) else 'w') as file:

        sys.stdout = file
        # Print the text
        if time_stamp:
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'\n',text)
    # Restore the original stdout
    sys.stdout = original_stdout


###########
#  Time  ######################################################################
###########
class Timer():

    def __init__(self):
        self.start_dt = None

    def start(self):
        self.start_dt = dt.datetime.now()

    def stop(self):
        end_dt = dt.datetime.now()
        print('Time taken: %s' % (end_dt - self.start_dt))
        



def result_dataset(result_path):
    data = pd.read_csv(result_path+'confMatrix_log.csv',index_col='Unnamed: 0')
    data = data.astype({'TN_train':'float',
         'FP_train':'float',
         'FN_train':'float',
         'TP_train':'float'})
    
    data['sum_val'] = data['TN_val']+data['FN_val']+data['FP_val']+data['TP_val']
    data['sum_train'] = data['TN_train']+data['FN_train']+data['FP_train']+data['TP_train']

    data['acc%_val'] = 100*(data['TN_val'] + data['TP_val']) / data['sum_val']
    data['precision_val'] = data['TP_val']/(data['TP_val']+data['FP_val']) 
    data['recall_val'] = data['TP_val']/(data['TP_val']+data['FN_val'])
    data['f1-score_val'] = 2*(data['precision_val']*data['recall_val'])/(data['precision_val']+data['recall_val'])

    data['acc%_train'] = 100*(data['TN_train'] + data['TP_train']) / data['sum_train']
    data['precision_train'] = data['TP_train']/(data['TP_train']+data['FP_train']) 
    data['recall_train'] = data['TP_train']/(data['TP_train']+data['FN_train'])
    data['f1-score_train'] = 2*(data['precision_train']*data['recall_train'])/(data['precision_train']+data['recall_train'])
    
    data['MCC_train'] = (data['TP_train']*data['TN_train']-data['FP_train']*data['FN_train'])/(np.sqrt(
        (data['TP_train']+data['FP_train'])*
        (data['TP_train']+data['FN_train'])*
        (data['TN_train']+data['FP_train'])*
        (data['TN_train']+data['FN_train'])))
    
    
    data['MCC_val'] =(data['TP_val']*data['TN_val']-data['FP_val']*data['FN_val'])/(np.sqrt(
        (data['TP_val']+data['FP_val'])*
        (data['TP_val']+data['FN_val'])*
        (data['TN_val']+data['FP_val'])*
        (data['TN_val']+data['FN_val'])))
    
    data['PW_val'] = (data['TP_val']+data['FN_val'])/data['sum_val']
    data['PW_train'] = (data['TP_train']+data['FN_train'])/data['sum_train']
#     data = data.iloc[50:,:]
#     data = data.mean(axis=0)

    return data







#####################
#  Results & Plots  ######################################################################
#####################


def loss_acc_f1_plot(data, plot_title='', save_image=False, image_name='image.jpg'):
    fig, ax_l = plt.subplots(3,1, figsize=(20,15))
    
    fig.suptitle(plot_title, fontsize=16)
    
    
    l1 = ax_l[0].plot(data['loss_train'].values,'r*-',label='Train - Left Y-axis')
    l2 = ax_l[0].plot(data['loss_val'].values,'b.-',label='Validation - Left Y-axis')

    l3 = ax_l[1].plot(data['acc%_train'].values,'r*-',label='Train - Left Y-axis')
    l4 = ax_l[1].plot(data['acc%_val'].values,'b.-',label='Validation - Left Y-axis')

    l5 = ax_l[2].plot(data['f1-score_train'].values,'r*-',label='Train - Left Y-axis')
    l6 = ax_l[2].plot(data['f1-score_val'].values,'b.-',label='Validation - Left Y-axis')


    lns_0 = l1+l2
    labs_0 = [l.get_label() for l in lns_0]
    lns_1 = l3+l4
    labs_1 = [l.get_label() for l in lns_1]
    lns_2 = l5+l6
    labs_2 = [l.get_label() for l in lns_2]


    ax_l[0].legend(lns_0, labs_0, fontsize=15)
    ax_l[1].legend(lns_1, labs_1, fontsize=15)
    ax_l[2].legend(lns_2, labs_2, fontsize=15)


    ax_l[0].set_xlabel('Epoch', fontsize=15)
    ax_l[0].set_ylabel('Loss', fontsize=17)
    ax_l[1].set_xlabel('Epoch', fontsize=15)
    ax_l[1].set_ylabel('Accuracy%', fontsize=17)
    ax_l[2].set_xlabel('Epoch', fontsize=15)
    ax_l[2].set_ylabel('f1-score', fontsize=17)

    xt = [np.round(i,0)      for i in np.arange(0,105,5)]
    xl = [str(np.round(i,0)) for i in np.arange(0,105,5)]  
    ax_l[0].set_xticks(xt);ax_l[0].set_xticklabels(xl,rotation=0, fontsize = 12)
    ax_l[1].set_xticks(xt);ax_l[1].set_xticklabels(xl,rotation=0, fontsize = 12)
    ax_l[2].set_xticks(xt);ax_l[2].set_xticklabels(xl,rotation=0, fontsize = 12)

    # ax_r.set_xticks(xt);
#     ax_r.set_xticklabels(xl,rotation=0, fontsize = 12)
    # ax_r[0].set_ylim(0,0.0005)
    plt.subplots_adjust(top=0.95) 

    ax_l[0].grid();ax_l[1].grid();ax_l[2].grid()
    if save_image:
        mohsenutils.save_image(image_name)





#####################
#  Find csv files  ######################################################################
#####################
import os

def find_csv_files(root_path):
    csv_file_paths = []
    
    # Walk through the directory and its subdirectories
    for foldername, subfolders, filenames in os.walk(root_path):
        for filename in filenames:
            # Check if the file has .csv extension
            if filename.endswith('.csv'):
                file_path = os.path.join(foldername, filename)
                # Add the path to the list
                csv_file_paths.append(file_path)
    
    if csv_file_paths:
        print("List of paths to .csv files:")
        for path in csv_file_paths:
            print(path)
        return csv_file_paths
    else:
        print("No CSV files found in the specified directory and its subdirectories.")
        return None



# Find text from text file
def find_XYZ_from_txt(file_path, XYZ, char_before_after):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        # content = content.replace('\n', '').replace('\t', '')
        # print(content)
        index_xyz = content.strip().find(XYZ)
        if index_xyz != -1:
            return content[index_xyz-char_before_after[0] :index_xyz + char_before_after[1]]
        else:
            print(f"'{XYZ}' not found in '{file_path}'")
    return None
