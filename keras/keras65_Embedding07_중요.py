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

# padded_x = padded_x[:, :, 1:]


#2. modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM

model = Sequential()
########################임베딩1 ##############################
# model.add(Embedding(input_dim=31, output_dim=100, input_length=5)) #input_dim : 단어 사전의 갯수 (None, 5, 100)
# =================================================================
#  embedding (Embedding)       (None, 5, 100)            3100

#  lstm (LSTM)                 (None, 10)                4440

#  dense (Dense)               (None, 10)                110

#  dense_1 (Dense)             (None, 1)                 11

# =================================================================
# Total params: 7,661
# Trainable params: 7,661
# Non-trainable params: 0
# _________________________________________________________________


########################임베딩2 ##############################
# model.add(Embedding(input_dim=31, output_dim=100)) #input_length 작성하지 않아도 돌아감!
# =================================================================
#  embedding (Embedding)       (None, None, 100)         3100

#  lstm (LSTM)                 (None, 10)                4440

#  dense (Dense)               (None, 10)                110

#  dense_1 (Dense)             (None, 1)                 11

# =================================================================
# Total params: 7,661
# Trainable params: 7,661
# Non-trainable params: 0
# _________________________________________________________________

########################임베딩3 ##############################
# model.add(Embedding(input_dim=100, output_dim=100)) #input_dim 개수를 맞추지 않아도 돌아가긴 함. 내가 가지고 있는 단어사전의 갯수보다 적은 수를 넣게 되면 성능 저하됨.
#input_dim = 30 #디폴트
#input_dim = 20 #단어사전의 갯수보다 작을 때 : 연산량 줄어, 단어사전에서 임의로 빼 : 성능조금저하
#input_dim = 40 #단어사전의 갯수보다 작을 때 : 연산량 늘어, 임의의 랜덤 임베딩 생성 : 성능조금저하

########################임베딩4 ##############################
model.add(Embedding(31,100)) # 잘 돌아감
#model.add(Embedding(31, 100, 5)) #에러
# model.add(Embedding(31, 100, input_length=5)) #잘 돌아감
# model.add(Embedding(31, 100, input_length=6)) #에러
# model.add(Embedding(31, 100, input_length=1)) #잘 돌아감. 약수(1,5)는 돌아감. 2,3,4 는 안됨!

# =================================================================
#  embedding (Embedding)       (None, None, 100)         3100

#  lstm (LSTM)                 (None, 10)                4440

#  dense (Dense)               (None, 10)                110

#  dense_1 (Dense)             (None, 1)                 11

# =================================================================
# Total params: 7,661
# Trainable params: 7,661
# Non-trainable params: 0
# _________________________________________________________________

model.add(LSTM(10)) # (None, 10)
model.add(Dense(10))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# #3.compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(padded_x, labels, epochs=1000)

#4. predict
result = model.evaluate(padded_x, labels)
print('loss : ', result)
#loss :  [0.00011025903222616762, 1.0]
