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
print(x.shape, y.shape)
x = x.reshape(94,6,1)


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
model.add(LSTM(units=32, input_shape=(6,1), return_sequences=True)) # timesteps , features
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
es = EarlyStopping(monitor='loss', mode='min', patience=30, verbose=1, restore_best_weights=True)
model.fit(x, y, epochs=5000, batch_size=4, 
          verbose=3, callbacks=[es]
          )
#4. 평가, 예측
result = model.evaluate(x, y)
print('loss :', result)

y_pred = model.predict([[95,96,97,98,99,100]])
print('101 원함 결과 :', y_pred)

# loss : 0.023361142724752426
# 101을 원함 결과 : [[100.780495]]

#loss : 0.0021382379345595837
# 102 원함 결과 : [[101.48538]]

#loss : 0.012470455840229988
# 103 원함 결과 : [[101.80302]]

#loss : 0.31337040662765503
# 104 원함 결과 : [[102.50407]]


# 6,1
# loss : 0.07093201577663422
# 101 원함 결과 : [[99.8805]]