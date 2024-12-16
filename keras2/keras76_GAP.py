import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import tensorflow as tf
from tensorflow.keras.applications import VGG16

tf.random.set_seed(333)
np.random.seed(333)
print(tf.__version__)   # 2.7.4

import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D #이미지 작업 Convolution2D == Conv2D (1D : 선, 3D : 입체형태)
from tensorflow.keras.layers import Flatten, Dropout, MaxPool2D, BatchNormalization, GlobalAveragePooling2D
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical


vgg16 = VGG16(#weights='imagenet',
              include_top=False,
              #input_shape=(32,32,3)
)
vgg16.trainable = True # false : 가중치 동결


#2. modeling
model=Sequential()
model.add(vgg16)
#model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))
model.summary()

