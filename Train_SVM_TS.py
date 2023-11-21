import sys
sys.path.append('./src/')
import numpy as np
import HazardNet_DataGen
import HazardNet_Utils as utils
import sklearn
from sklearn import metrics
from sklearn import svm
from sklearn.preprocessing import RobustScaler

root_path = './TRAIN_RESULTS/SVM/'
res_folder_name = ''
readme_text = """ 
SVM Model for time sepereated train and test data.
"""
print_text_path = './TRAIN_RESULTS/SVM/SVM_Paper.txt'



# Data generator
OneD_data = HazardNet_DataGen.OneD_DataGen(model_type=['prediction','1D_Inlet'], 
                                           compute_server='lab_gpu', 
                                           STIT=0.05, 
                                           data_period=None) 
X, lbe = OneD_data.load_data()

X_train, lbe_train = X.loc['2019-05-01 00:00:00':'2019-06-01 00:00:00'].values, lbe.loc['2019-05-01 00:00:00':'2019-06-01 00:00:00'].room_label.values
X_val, lbe_val =     X.loc['2019-06-01 00:00:00':'2019-06-08 00:00:00'].values, lbe.loc['2019-06-01 00:00:00':'2019-06-08 00:00:00'].room_label.values





# Standardize the data
transformer = RobustScaler()
transformer = transformer.fit(X_train)

X_train_tf = transformer.transform(X_train)
X_val_tf =  transformer.transform(X_val)



# # [Samples, time_steps, number of featuers]
# utils.prnt_w2f(f'X_train_tf {X_train_tf.shape} lbe_train {lbe_train.shape}', print_text_path)
# utils.prnt_w2f(f'X_val_tf {X_val_tf.shape} lbe_val {lbe_val.shape}', print_text_path)
# classifier = svm.SVC(kernel='rbf')
# # classifier = svm.SVC(kernel='linear')
# classifier.fit(X_train_tf, lbe_train)
# lbe_predict = classifier.predict(X_val_tf)
# print(metrics.classification_report(lbe_val, lbe_predict))
# print(lbe_val.sum())
# print(metrics.f1_score(lbe_val, lbe_predict))


def _create_dataset(X, y):
    TW = 36
    Xs, ys = [],[]
    for i in range(len(X)-TW):
        Xs.append(X[i:i+TW])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


X_train_tf_3d, lbe_train = _create_dataset(X_train_tf, lbe_train)
X_val_tf_3d, lbe_val = _create_dataset(X_val_tf, lbe_val)

X_train_tf_2d = X_train_tf_3d.reshape(X_train_tf_3d.shape[0],X_train_tf_3d.shape[1]*X_train_tf_3d.shape[2])
X_val_tf_2d = X_val_tf_3d.reshape(X_val_tf_3d.shape[0],X_val_tf_3d.shape[1]*X_val_tf_3d.shape[2])

# [Samples, time_steps, number of featuers]
utils.prnt_w2f(f'X_train_tf_3d {X_train_tf_3d.shape} lbe_train {lbe_train.shape}', print_text_path)
utils.prnt_w2f(f'X_val_tf_3d {X_val_tf_3d.shape} lbe_val {lbe_val.shape}', print_text_path)


utils.prnt_w2f(f'X_train_tf_2d {X_train_tf_2d.shape} lbe_train {lbe_train.shape}', print_text_path)
utils.prnt_w2f(f'X_val_tf_2d {X_val_tf_2d.shape} lbe_val {lbe_val.shape}', print_text_path)



classifier = svm.SVC(kernel='rbf')
# classifier = svm.SVC(kernel='linear')
classifier.fit(X_train_tf_2d, lbe_train)
lbe_predict = classifier.predict(X_val_tf_2d)
print(metrics.classification_report(lbe_val, lbe_predict))
print(metrics.f1_score(lbe_val, lbe_predict))
print((classifier.coef_).shape)