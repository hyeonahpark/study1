import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16

vgg16 = VGG16(weights = 'imagenet', include_top=False, # fully_connected layer 삭제, 그러면 input shape 변경 가능!
              input_shape=(32,32,3))


model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

model.trainable = False 
model.summary()

print(model.weights)
print(len(model.weights))
print(len(model.trainable_weights))


import pandas as pd
pd.set_option('max_colwidth', None)
layers=[(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
print(results)