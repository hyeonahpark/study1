from keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

#1. data
docs = [
    '너무 재미있다', '참 최고에요', '참 잘만든 영화에요',
    '추천하고 싶은 영화입니다.', '한 번 더 보고 싶어요.', '글쎄',
    '별로에요', '생각보다 지루해요', '연기가 어색해요',
    '재미없어요', '너무 재미없다.', '참 재밌네요.',
    '준영이 바보', '반장 잘생겼다', '태운이 또 구라친다'
]

labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
#{'참': 1, '너무': 2, '재미있다': 3, '최고에요': 4, '잘만든': 5, '영화에요': 6, '추천하고': 7, '싶은': 8, '영화입니다': 9, '한': 10, '번': 11, '더': 12, '보고': 13, '싶어요': 14, '글쎄': 15, '별로에요': 16, '생
# 각보다': 17, '지루해요': 18, '연기가': 19, '어색해요': 20, '재미없어요': 21, '재미없다': 22, '재밌네요': 23, '준영이': 24, '바보': 25, '반장': 26, '잘생겼다': 27, '태운이': 28, '또': 29, '구라친다': 30}

x = token.texts_to_sequences(docs)
print(x)
print(type(x)) #<class 'list'>
padded_x = pad_sequences(x, padding = 'pre', #'post'
                        #  maxlen=5,
                         #truncating='pre', #'post'
                         )

print(padded_x)
print(padded_x.shape) #(15, 5)

x_pred = ['태운이 참 재미없다']
# token2 = Tokenizer()
# print(token2.word_index)
# print(token2.word_index) #{'태': 1, '운': 2, '이': 3, '참': 4, '재': 5, '미': 6, '없': 7, '다': 8}
x_pred = token.texts_to_sequences(x_pred)

#x_pred2 : [[28, 1, 22]]

padded_x2 = pad_sequences(x_pred, padding = 'pre', #'post'
                         maxlen=5,
                         #truncating='pre', #'post'
                         )
print(padded_x2)
#[[ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#    0  0  0 28  1 22]]
print(padded_x2.shape) # (1, 30)

# padded_x2 = padded_x2.reshape(1,30,1)
# print(padded_x2.shape) # (1, 30, 1)

from tensorflow.keras.utils import to_categorical
padded_x= to_categorical(padded_x)
padded_x2 = to_categorical(padded_x2, num_classes=31)
# padded_x = padded_x[:, :, 1:]

print(padded_x)
print(padded_x.shape) # (15, 5, 31) 
print(padded_x2.shape) #(1, 31, 5)

padded_x2 = padded_x2.reshape(1,5,31)

print(padded_x2.shape) #(1, 5, 31)


#2. modeling 
#######DNN 맹글기######
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Embedding, LSTM
from sklearn.model_selection import train_test_split
import time
# # 단어 임베딩 함수의 입력 인덱스 수
# word_size = len(token.word_index) + 1
x_train, x_test, y_train, y_test = train_test_split(padded_x, labels, train_size=0.9, random_state=1186)

model = Sequential()
# model.add(Embedding(word_size, 8, input_length=5))
# model.add(Flatten())
model.add(LSTM(32,input_shape=(5,31), return_sequences=True))
model.add(LSTM(16))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

start_time = time.time()
#3. compile
model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1000, batch_size=8)
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
print('태운이 참 재미없다 :', np.round((y_pred)))


# Accuracy : 1.0000
# 1/1 [==============================] - 0s 13ms/step - loss: 7.4030e-04 - accuracy: 1.0000
# loss : 0.0007403029594570398
# 태운이 참 재미없다 : [[0.]]