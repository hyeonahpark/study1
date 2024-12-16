import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16

model = VGG16(weights = 'imagenet', include_top=True, # fully_connected layer 삭제, 그러면 input shape 변경 가능!
              #input_shape=(224,224,3)
              )


model.summary()

import pandas as pd
pd.set_option('max_colwidth', None)
layers=[(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
print(results)