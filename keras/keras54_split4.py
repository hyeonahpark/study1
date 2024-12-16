#54-3 copy
#(N, 10, 1) -> (N, 5, 2)
# 맹그러바
import numpy as np

a = np.array(range(1, 101))

x_predict = np.array(range(96, 106))  #101부터 107까지 찾아라

size = 7


def split_x(dataset, size) :
    aaa=[]
    for i in range(len(dataset)-size+1):
        subset = dataset[i: (i+size)]
        aaa.append(subset)
    return np.array(aaa)


bbb = split_x(a, size)
# print(bbb)
# print(bbb.shape)

x=bbb[:, :-1]
y=bbb[:, -1]
# print(x)
# print(y)
# print(x.shape, y.shape) #(94, 6) (94,)
x = x.reshape(94,3,2)
# print(x.shape) # (94,3,2)

# print(a)
#[  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18
#   19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36
#   37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53  54
#   55  56  57  58  59  60  61  62  63  64  65  66  67  68  69  70  71  72
#   73  74  75  76  77  78  79  80  81  82  83  84  85  86  87  88  89  90
#   91  92  93  94  95  96  97  98  99 100]
# print(x_predict)
#[ 96  97  98  99 100 101 102 103 104 105]

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

#2. 모델 구성
model = Sequential()
model.add(LSTM(units=32, input_shape=(3,2), return_sequences=True)) # timesteps , features
model.add(LSTM(32, return_sequences=True)) # timesteps , features
model.add(LSTM(32))
# Flaten 사용하는 방법도 있음 
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

# model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='loss', mode='min', patience=500, verbose=1, restore_best_weights=True)
model.fit(x, y, epochs=3000, batch_size=4, 
          verbose=1, callbacks=[es]
          )

#4. 평가, 예측
result = model.evaluate(x, y)
print('loss :', result)

x_pred=np.array([101, 102, 103, 104, 105, 106]).reshape(1,3,2) 

y_pred = model.predict(x_pred)
print('107 원함 결과 :', y_pred)


#54-3
# loss : 0.07093201577663422
# 101 원함 결과 : [[99.8805]]

#54-4
# loss : 0.21079370379447937
# 101 원함 결과 : [[100.502365]]

#loss : 0.48613131046295166
# 102 원함 결과 : [[101.35778]]

# loss : 0.003742365865036845
# 103 원함 결과 : [[102.38874]]

# loss : 0.006765642669051886
# 104 원함 결과 : [[103.03755]]

# loss : 0.022582199424505234
# 105 원함 결과 : [[103.468735]]

# loss : 0.0234188474714756
# 106 원함 결과 : [[104.49903]]

# loss : 0.0010615518549457192
# 107 원함 결과 : [[104.92213]]
