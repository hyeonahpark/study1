import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16

vgg16 = VGG16(weights = 'imagenet', include_top=False, # fully_connected layer 삭제, 그러면 input shape 변경 가능!
              input_shape=(32,32,3))

vgg16.trainable = False  

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

#model.trainable = False 
model.summary()

print(model.weights)
print(len(model.weights))
print(len(model.trainable_weights))


"""
                              trainable = True    // model = False  // VGG False
len(model.weights)                   30           //      30        //    30                                 
len(model.trainable_weights)         30           //      0        //      4      

"""