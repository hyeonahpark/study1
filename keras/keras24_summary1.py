from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np



#2. modeling
model= Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))

model.summary() #모델 계층 보여줌 print 안써도 됨 !
#((input 노드 수)+(bias(1)))*다음노드 수 = param #
#input 노드수*다음노드 수 + 다음노드 수
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  dense (Dense)               (None, 3)                 6

#  dense_1 (Dense)             (None, 4)                 16

#  dense_2 (Dense)             (None, 3)                 15

#  dense_3 (Dense)             (None, 1)                 4

# =================================================================
# Total params: 41
# Trainable params: 41
# Non-trainable params: 0
# _________________________________________________________________

