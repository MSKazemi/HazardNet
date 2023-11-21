import sys
sys.path.append('./src/')
import numpy as np
from matplotlib import pyplot as plt
import HazardNet_DataGen
import HazardNet_Utils as utils
import random
import numpy as np
import tensorflow as tf
SEED = 4321
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

root_path = './TRAIN_RESULTS/LSTM/TEST/LSTM_1D_Inlet/'
res_folder_name = ''
readme_text = """ 
LSTM Model for time separated data.
"""
result_path = root_path
print_text_path = './TRAIN_RESULTS/LSTM/TEST/LSTM_1D_Inlet/LSTM_1D_Inlet.txt'



# Set the random seeds

import sys
sys.path.append('./src/')
import numpy as np
from matplotlib import pyplot as plt
import HazardNet_DataGen
import sklearn
from sklearn.preprocessing import RobustScaler
import keras
from keras.models import Sequential, load_model
import tensorflow as tf
import HazardNet_Utils as utils

def LSTM_create_dataset(X, y):
    TW = 36
    Xs, ys = [],[]
    for i in range(len(X)-TW):
        Xs.append(X[i:i+TW])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)



# Data generator
OneD_data = HazardNet_DataGen.OneD_DataGen(model_type=['prediction','1D_Inlet'], 
                                           compute_server='lab_gpu', 
                                           STIT=0.05, 
                                           data_period=None) 
X, lbe = OneD_data.load_data()

X_train, lbe_train = X.loc['2019-05-01 00:00:00':'2019-06-01 00:00:00'].values, lbe.loc['2019-05-01 00:00:00':'2019-06-01 00:00:00'].room_label.values
X_val, lbe_val =     X.loc['2019-06-01 00:00:00':'2019-06-08 00:00:00'].values, lbe.loc['2019-06-01 00:00:00':'2019-06-08 00:00:00'].room_label.values






transformer = RobustScaler()
transformer = transformer.fit(X_train)

X_train_tf = transformer.transform(X_train)
X_val_tf =  transformer.transform(X_val)

X_train_tf_2d, lbe_train = LSTM_create_dataset(X_train_tf, lbe_train)
X_val_tf_2d, lbe_val = LSTM_create_dataset(X_val_tf, lbe_val)


# [Samples, time_steps, number of featuers]
utils.prnt_w2f(f'X_train_tf_2d {X_train_tf_2d.shape} lbe_train {lbe_train.shape}', print_text_path)
utils.prnt_w2f(f'X_val_tf {X_val_tf.shape} lbe_val {lbe_val.shape}', print_text_path)


# tf.keras.backend.set_floatx('float64')
# with tf.device(tf.DeviceSpec(device_type='GPU',device_index=1)):
model = Sequential()
model.add(keras.layers.LSTM(units=32,dropout=0.4,input_shape=(X_train_tf_2d.shape[1], X_train_tf_2d.shape[2]),return_sequences=True))
model.add(keras.layers.Dropout(rate=0.4))
model.add(keras.layers.LSTM(units=32,dropout=0.4))
model.add(keras.layers.Dropout(rate=0.4))
# Add the output layer with a sigmoid activation function for binary classification
model.add(keras.layers.Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics='accuracy')

history = model.fit(
    X_train_tf_2d, 
    lbe_train,
    epochs=100, 
    batch_size=200, 
    validation_split=0.1, 
    shuffle=False
    )

lbe_pred = model.predict(X_val_tf_2d)
lbe_pred = (lbe_pred > 0.5).astype(int)
lbe_pred_inv = lbe_pred.reshape(1,-1)

utils.prnt_w2f(sklearn.metrics.confusion_matrix(lbe_val.flatten(),lbe_pred_inv.flatten()), print_text_path)
utils.prnt_w2f(sklearn.metrics.f1_score(lbe_val.flatten(),lbe_pred_inv.flatten()),print_text_path)
