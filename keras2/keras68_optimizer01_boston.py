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
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=3333)

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
# learning_rate = 0.01
# learning_rate = 0.001 #default 값
# learning_rate = 0.005 
# learning_rate = 0.0001 # 박살 !
learning_rate = 0.0007 #0.6375050183176865
# learning_rate = 0.0008 #0.6373160739385821
# learning_rate = 0.0009 #0.631=70128630491578

model.compile(loss = 'mse', optimizer = Adam(learning_rate=learning_rate))
model.fit(x_train, y_train, validation_split=0.2, epochs = 100, batch_size=32)

#4. predict
loss = model.evaluate(x_test, y_test, verbose=0)
print('lr:{0}, loss:{1}'.format(learning_rate, loss))

y_predict = model.predict(x_test, verbose=0)
r2 = r2_score(y_test, y_predict)
print('lr:{0}, r2 :{2}'.format(learning_rate, r2))

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

########################################[실습]#########################################
# lr = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]

#10_fetch_covtype
#12_kaggle_santander
#13_kaggle_otto
#14_mnist
#16_cifar10
#17_cifar100
#18_kaggle_cat_dog
#19_horse
#21_jena