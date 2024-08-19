import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras.utils import to_categorical

text = '나는 지금 진짜 진짜 매우 매우 맛있는 김밥을 엄청 마구 마구 마구 마구 먹었다.'

token = Tokenizer()
token.fit_on_texts([text])

print(token.word_index)  # 1. 많이 나오는 순서, 2. 먼저 나오는 순서
#{'마구': 1, '진짜': 2, '매우': 3, '나는': 4, '지금': 5, '맛있는': 6, '김밥을': 7, '엄청': 8, '먹었다': 9}
print(token.word_counts)
#OrderedDict([('나는', 1), ('지금', 1), ('진짜', 2), ('매우', 2), ('맛있는', 1), ('김밥을', 1), ('엄청', 1), ('마구', 4), ('먹었다', 1)])

x = token.texts_to_sequences([text])
print(x) # [[4, 5, 2, 2, 3, 3, 6, 7, 8, 1, 1, 1, 1, 9]]
# print(x.shape) #리스트는 shape 없음

####### 원핫 3가지 맹글기 #########
x_array = np.array(x).reshape(-1, 1)

##oneHot 1-1
# x = to_categorical(x_array)
# x = x[:,1:]
# print(x)
# [[0. 0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 1. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 1.]]
# print(x.shape) # (14, 9)


# ##oneHot 1-2
# x_flatten = [item for sublist in x for item in sublist]
# print(x_flatten)
# pandas Series로 변환
# x_series = pd.Series(x_flatten)
# get_dummies로 원핫 인코딩
# x = pd.get_dummies(pd.Series(np.array(x).reshape(-1,)))

# x = pd.get_dummies(sum(x, []))
x = np.array(x).reshape(-1,)
x = pd.get_dummies(x)
# x = x.to_numpy()
print(x)
#0   0  0  0  1  0  0  0  0  0
# 1   0  0  0  0  1  0  0  0  0
# 2   0  1  0  0  0  0  0  0  0
# 3   0  1  0  0  0  0  0  0  0
# 4   0  0  1  0  0  0  0  0  0
# 5   0  0  1  0  0  0  0  0  0
# 6   0  0  0  0  0  1  0  0  0
# 7   0  0  0  0  0  0  1  0  0
# 8   0  0  0  0  0  0  0  1  0
# 9   1  0  0  0  0  0  0  0  0
# 10  1  0  0  0  0  0  0  0  0
# 11  1  0  0  0  0  0  0  0  0
# 12  1  0  0  0  0  0  0  0  0
# 13  0  0  0  0  0  0  0  0  1
print(x.shape) #(14,9)


# ##oneHot 1-3
# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder(sparse=False) #sparse=True가 기본값
# x = ohe.fit_transform(x_array)

# print(x)
#[[0. 0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 1. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 1.]]
# print(x.shape) #(14,9)

