import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. data
x_data = np.array([[0,0],[0,1],[1,0],[1,1]])
y_data = np.array([0,1,1,0])
print(x_data.shape, y_data.shape) #(4, 2) (4,)

#2. model
# model = LinearSVC()
# model = Perceptron()

model = Sequential()
model.add(Dense(10, input_dim=2, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. training
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, batch_size=1, epochs=100)

#4. predict
# acc = model.score(x_data, y_data)
# print('model.score : ', acc)

loss=model.evaluate(x_data, y_data)
print('acc : ', loss[1])

y_predict = np.round(model.predict(x_data)).reshape(-1,).astype(int)
acc2 = accuracy_score(y_data, y_predict)
print('accuracy score : ', acc2)
print('=================================')
print(y_data) #[0 1 1 1]
print(y_predict)#[0 1 1 1]


# acc :  1.0
# accuracy score :  1.0
# =================================
# [0 1 1 0]
# [0 1 1 0]