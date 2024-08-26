# 15개의 행에서 5개를 더 넣어서 맹글기

from keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

#1. data
docs = [
    '너무 재미있다', '참 최고에요', '참 잘만든 영화에요',
    '추천하고 싶은 영화입니다.', '한 번 더 보고 싶어요.', '글쎄',
    '별로에요', '생각보다 지루해요', '연기가 어색해요',
    '재미없어요', '너무 재미없다.', '참 재밌네요.',
    '준영이 바보', '반장 잘생겼다', '태운이 또 구라친다', 
    '난 용기가 없어', '누리는 똑똑하다', '공부하기 싫어요', '나는 오늘 너무 집에 가고 싶어요', '오늘은 밥이 정말 너무 맛이 없어요.'
]

labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0,0,1,0,1,0])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
# {'너무': 1, '참': 2, '싶어요': 3, '재미있다': 4, '최고에요': 5, '잘만든': 6, '영화에요': 7, '추천하고': 8, '싶은': 9, '영화입니다': 10, '한': 11, '번': 12, '더': 13, '보고': 14, '글쎄': 15, 
# '별로에요': 16, '생각보다': 17, '지루해요': 18, '연기가': 19, '어색해요': 20, '재미없어요': 21, '재미없다': 22, '재밌네요': 23, '준영이': 24, '바보': 25, '반장': 26, '잘생겼다': 27, '태운이': 28, '또': 29, '구라친다': 30, '난': 31, '용기가': 32, '없어': 33, '누리는': 34, '똑똑하다': 35, '공부하기': 36, '싫어요': 37, '나는': 38, '오늘': 39, '집에': 40, '가고': 41, '오늘은': 42, 
# '밥이': 43, '정말': 44, '맛이': 45, '없어요': 46}

x = token.texts_to_sequences(docs)
print(x)
print(type(x)) #<class 'list'>
padded_x = pad_sequences(x, padding = 'pre', #'post'
                         maxlen=5,
                         truncating='pre', #'post'
                         )
print(padded_x)
print(padded_x.shape) #(20, 5)



x_pred = ['누리는 용기가 없어']
# token2 = Tokenizer()
# print(token2.word_index)
# print(token2.word_index) #{'태': 1, '운': 2, '이': 3, '참': 4, '재': 5, '미': 6, '없': 7, '다': 8}
x_pred = token.texts_to_sequences(x_pred)

#x_pred2 : [[28, 1, 22]]

padded_x2 = pad_sequences(x_pred, padding = 'pre', #'post'
                         maxlen=5
                         #truncating='pre', #'post'
                         )
print(padded_x2)
# [[ 0 38 40 41  3]]
print(padded_x2.shape) # (1, 5)
# padded_x2 = padded_x2.reshape(1,30,1)
# print(padded_x2.shape) # (1, 30, 1)

##원핫 2
# padded_x = np.array(padded_x).reshape(-1,)
# padded_x = pd.get_dummies(padded_x)
# padded_x2 = np.array(padded_x2).reshape(-1,)
# padded_x2 = pd.get_dummies(padded_x2, dummy_na=31)

##원핫 3
from tensorflow.keras.utils import to_categorical
padded_x= to_categorical(padded_x)
padded_x2 = to_categorical(padded_x2, num_classes=47)

# print(padded_x2)
print(padded_x.shape) # (20, 5, 47) 
print(padded_x2.shape) #(1, 5, 47)


#2. modeling 
#######DNN 맹글기######
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Embedding, LSTM, Conv1D
from sklearn.model_selection import train_test_split
import time
# # 단어 임베딩 함수의 입력 인덱스 수
# word_size = len(token.word_index) + 1
x_train, x_test, y_train, y_test = train_test_split(padded_x, labels, train_size=0.9, random_state=1186)

model = Sequential()
model.add(Conv1D(30, (3), input_shape=(5, 47), padding='same')) 
model.add(Conv1D(32, (2),  padding='same')) 
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

start_time = time.time()
#3. compile
model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1000, batch_size=4)
end_time = time.time()
model.fit(padded_x, labels, epochs = 1000)

# model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
# model.fit(padded_x, labels, epochs=1000)
print("\nAccuracy : %.4f" % (model.evaluate(padded_x, labels)[1]))

#4. predict
result = model.evaluate(x_test, y_test)
print('loss :', result[0])

# print(x_pred.shape)

y_pred = model.predict(padded_x2)
print('누리는 용기가 없어 :', np.round((y_pred)))


# Accuracy : 1.0000
# 1/1 [==============================] - 0s 10ms/step - loss: 7.2699e-06 - accuracy: 1.0000
# loss : 7.269909019669285e-06
# 나는 집에 가고 싶어요 : [[1.]]
 
 
# Accuracy : 1.0000
# 1/1 [==============================] - 0s 11ms/step - loss: 7.3768e-07 - accuracy: 1.0000
# loss : 7.376824555649364e-07
# 누리는 용기가 없어 : [[0.]]