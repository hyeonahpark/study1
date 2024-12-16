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
#[[2, 3], [1, 4], [1, 5, 6], [7, 8, 9], [10, 11, 12, 13, 14], [15], [16], [17, 18], [19, 20], [21], [2, 22], [1, 23], [24, 25], [26, 27], [28, 29, 30]]
print(type(x)) #<class 'list'>
padded_x = pad_sequences(x, padding = 'pre', #'post'
                        #  maxlen=5,
                         #truncating='pre', #'post'
                         )
# padding='post'를 하면 0이 뒤에 채워지고, padding='pre'를 하면 0이 앞에 채워짐. default값은 pre, 최대길이도 default값임.
#max_len 길이를 정해주고, 그 길이만큼 앞에서부터 자름. truncating = 'post'를 하면 뒤에서부터 자름
print(padded_x)
print(padded_x.shape) #(15, 5)
#[[ 0  0  0  2  3]
#  [ 0  0  0  1  4]
#  [ 0  0  1  5  6]
#  [ 0  0  7  8  9]
#  [10 11 12 13 14]
#  [ 0  0  0  0 15]
#  [ 0  0  0  0 16]
#  [ 0  0  0 17 18]
#  [ 0  0  0 19 20]
#  [ 0  0  0  0 21]
#  [ 0  0  0  2 22]
#  [ 0  0  0  1 23]
#  [ 0  0  0 24 25]
#  [ 0  0  0 26 27]
#  [ 0  0 28 29 30]]
x_pred = ['태운이 참 재미없다.']
token2 = Tokenizer()
token2.fit_on_texts(x_pred)
# print(token2.word_index)
# print(token2.word_index) #{'태': 1, '운': 2, '이': 3, '참': 4, '재': 5, '미': 6, '없': 7, '다': 8}
x_pred2 = token.texts_to_sequences(x_pred)

#x_pred2 : [[28, 1, 22]]

padded_x2 = pad_sequences(x_pred2, padding = 'pre', #'post'
                         maxlen=5,
                         #truncating='pre', #'post'
                         )

# print(padded_x2) #[[ 0  0 28  1 22]]

padded_x = padded_x.reshape(15,5,1)
padded_x2 = padded_x2.reshape(1,5,1)


#2. modeling 
#######DNN 맹글기######
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Embedding, LSTM
from sklearn.model_selection import train_test_split
import time
# # 단어 임베딩 함수의 입력 인덱스 수
word_size = len(token.word_index) + 1
x_train, x_test, y_train, y_test = train_test_split(padded_x, labels, train_size=0.9, random_state=1186)

model = Sequential()
# model.add(Embedding(word_size, 8, input_length=5))
# model.add(Flatten())
model.add(LSTM(10,input_shape=(5,1), return_sequences=True))
model.add(LSTM(10))
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


# Accuracy : 0.9333
# 1/1 [==============================] - 0s 13ms/step - loss: 0.4025 - accuracy: 0.5000
# loss : 0.40252217650413513
# 태운이 참 재미없다 : [[0.]]