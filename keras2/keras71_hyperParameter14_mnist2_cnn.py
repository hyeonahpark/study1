import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Flatten
from sklearn.model_selection import train_test_split
import warnings
from sklearn.model_selection import RandomizedSearchCV
warnings.filterwarnings('ignore')
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import pandas as pd

#1. data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train=x_train/255.
x_test=x_test/255.

# print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

y_train=pd.get_dummies(y_train)
y_test=pd.get_dummies(y_test)

#2. model
def build_model(drop=0.5, optimizer='adam', activation='relu',
                node1=128, node2=64, node3=32, node4=16, node5=8, lr=0.005):
    inputs= Input(shape=(28,28, 1), name='input1')
    x=Conv2D(node1, (3,3), activation=activation, name='hidden1')(inputs)
    x=Dropout(drop)(x)
    x=Conv2D(node2, (3,3), activation=activation, name='hidden2')(x)
    x=Dropout(drop)(x)
    x=Conv2D(node3, (3,3), activation=activation, name='hidden3')(x)
    x=Dropout(drop)(x)
    x=Flatten()(x)
    x=Dense(node4, activation=activation, name='hidden4')(x)
    x=Dense(node5, activation=activation, name='hidden5')(x)
    outputs = Dense(10, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=optimizer, metrics=['mae'], loss = 'categorical_crossentropy') 
    return model


def create_hyperparameter():
    batchs = [64, 16, 8]
    optimizers = ['adam', 'rmsprop', 'adadelta']
    dropouts = [0.2, 0.3]
    activations = ['relu', 'elu', 'selu', 'linear']
    node1=[128, 64, 32, 16]
    node2=[128, 64, 32, 16]
    node3=[128, 64, 32, 16]
    node4=[128, 64, 32, 16]
    node5=[128, 64, 32, 16, 8]
    lr = [0.1, 0.001, 0.005, 0.01]
    # epochs = [1000]  # Iteration을 위한 epochs 추가
    return {'batch_size' : batchs,
            'optimizer' : optimizers,
            'drop':dropouts,
            'activation':activations,
            'node1':node1,
            'node2':node2,
            'node3':node3,
            'node4':node4,
            'node5':node5,  
            'lr' :lr 
            # 'epochs': epochs  # 하이퍼파라미터에 epochs 추가         
}

hyperparameter = create_hyperparameter()
print(hyperparameter)

# 4. Wrapping the model using KerasRegressor
keras_model = KerasClassifier(build_fn=build_model, verbose=1)

model = RandomizedSearchCV(keras_model, hyperparameter, cv=2,
                           n_iter=1,
                           #n_jobs=-1,
                           verbose=1)

import time
start_time=time.time()

import datetime

date = datetime.datetime.now()
date = date.strftime("%y%m%d_%H%M%S") # 240726_165505

PATH_SAVE = 'C:\\ai5\\_save\\keras71\\k71_14_1\\'

filename = '{epoch:04d}-{val_loss:.8f}.hdf5'
filepath = ''.join([PATH_SAVE, 'newDrug_', date, "_", filename])
#################### mcp 세이브 파일명 만들기 끝 ###################
mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    verbose = 1,
    save_best_only = True,
    filepath = filepath
)

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 20,
    restore_best_weights = True
)

rlr = ReduceLROnPlateau(
    monitor = 'val_loss',
    mode = 'auto',
    patience = 10,
    verbose = 1,
    factor = 0.9
)

model.fit(x_train, y_train, validation_split=0.1, epochs=1,
          callbacks = [es, mcp, rlr])

end_time=time.time()

from sklearn.metrics import accuracy_score, r2_score
y_predict = model.predict(x_test)
print("걸린 시간 :", round(end_time-start_time, 2))
print("model.best_params_ : ", model.best_params_)
print("model.best_scroe :", model.best_score_)
# print('r2_score : ', r2_score(y_test, y_predict))

# 걸린 시간 : 58.98
# model.best_params_ :  {'optimizer': 'adam', 'node5': 32, 'node4': 16, 'node3': 32, 'node2': 32, 'node1': 128, 'drop': 0.4, 'batch_size': 16, 'activation': 'linear'}
# model.best_scroe : -3122.6153157552085
# r2_score :  0.4065448847689277

# model.best_params_ :  {'optimizer': 'rmsprop', 'node5': 128, 'node4': 128, 'node3': 32, 'node2': 32, 'node1': 128, 'lr': 0.001, 'drop': 0.2, 'batch_size': 8, 'activation': 'linear'}