from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
import tensorflow as tf 
import random as rn
rn.seed(337)
tf.random.set_seed(337) # seed 고정
np.random.seed(337)

#1. data
datasets = load_boston()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=337)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. modeling
model = Sequential()
model.add(Dense(10, input_dim = 13))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. compile
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', mode = 'min', patience=30, verbose=1, restore_best_weights=True)

rlr = ReduceLROnPlateau(monitor='val_loss', mode='auto', patience=25, verbose=1, factor=0.8) #factor는 곱하기! 

learning_rate = 0.005 #default : 0.001
model.compile(loss = 'mse', optimizer = Adam(learning_rate=learning_rate))
model.fit(x_train, y_train, validation_split=0.2, epochs = 1000, batch_size=32, callbacks=[es, rlr])

#4. predict
print("================기본출력===============")
loss = model.evaluate(x_test, y_test, verbose=0)
print('lr:{0}, loss:{1}'.format(learning_rate, loss))

y_predict = model.predict(x_test, verbose=0)
r2 = r2_score(y_test, y_predict)
print('lr:{0}, r2 :{1}'.format(learning_rate, r2))

#loss:34.11334991455078 #learning rate 0.001 #default
# r2 :0.6366904657062408

# loss:34.697418212890625 #learning rate 0.01
# r2 :0.6304700624761163

# loss:350.7092590332031 #learning rate 0.0001
# r2 :-2.7350779800147946

# loss:34.42169952392578 #learning rate 0.005
# r2 :0.6334065116807991

# loss:34.083072662353516 #learning rate 0.0009
# r2 :0.6370128778917529

# loss:34.036861419677734 #learning rate 0.0007
# r2 :0.6375050183176865

###########################################################################

# ================기본출력===============
# lr:0.005, loss:33.42071533203125
# lr:0.005, r2 :0.6440670694279149