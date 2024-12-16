import numpy as np
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
import warnings
from sklearn.model_selection import RandomizedSearchCV
warnings.filterwarnings('ignore')
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


#1. data
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=5, train_size=0.9,
                                                    )

# print(x_train.shape, y_train.shape)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler=StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)


#2. model
def build_model(drop=0.5, optimizer='adam', activation='relu',
                node1=128, node2=64, node3=32, node4=16, node5=8, lr=0.001):
    inputs= Input(shape=(10, ), name='input1')
    x=Dense(node1, activation=activation, name='hidden1')(inputs)
    x=Dropout(drop)(x)
    x=Dense(node2, activation=activation, name='hidden2')(x)
    x=Dropout(drop)(x)
    x=Dense(node3, activation=activation, name='hidden3')(x)
    x=Dropout(drop)(x)
    x=Dense(node4, activation=activation, name='hidden4')(x)
    x=Dense(node5, activation=activation, name='hidden5')(x)
    outputs = Dense(1, activation='linear')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=optimizer, metrics=['mae'], loss = 'mse') 
    return model


def create_hyperparameter():
    batchs = [32, 16, 8, 1, 64]
    optimizers = ['adam', 'rmsprop', 'adadelta']
    dropouts = [0.2, 0.3, 0.4, 0.5]
    activations = ['relu', 'elu', 'selu', 'linear']
    node1=[128, 64, 32, 16]
    node2=[128, 64, 32, 16]
    node3=[128, 64, 32, 16]
    node4=[128, 64, 32, 16]
    node5=[128, 64, 32, 16, 8]
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
            # 'epochs': epochs  # 하이퍼파라미터에 epochs 추가         
}

hyperparameter = create_hyperparameter()
print(hyperparameter)

# 4. Wrapping the model using KerasRegressor
keras_model = KerasRegressor(build_fn=build_model, verbose=1)

model = RandomizedSearchCV(keras_model, hyperparameter, cv=3,
                           n_iter=5,
                           #n_jobs=-1,
                           verbose=1)

import time
start_time=time.time()

import datetime

date = datetime.datetime.now()
date = date.strftime("%y%m%d_%H%M%S") # 240726_165505

PATH_SAVE = 'C:\\ai5\\_save\\keras71\\k71_03\\'

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

model.fit(x_train, y_train, validation_split=0.1, epochs=500,
          callbacks = [es, mcp, rlr])

end_time=time.time()

from sklearn.metrics import accuracy_score, r2_score
y_predict = model.predict(x_test)
print("걸린 시간 :", round(end_time-start_time, 2))
print("model.best_params_ : ", model.best_params_)
print("model.best_scroe :", model.best_score_)
print('r2_score : ', r2_score(y_test, y_predict))

# 걸린 시간 : 58.98
# model.best_params_ :  {'optimizer': 'adam', 'node5': 32, 'node4': 16, 'node3': 32, 'node2': 32, 'node1': 128, 'drop': 0.4, 'batch_size': 16, 'activation': 'linear'}
# model.best_scroe : -3122.6153157552085
# r2_score :  0.4065448847689277