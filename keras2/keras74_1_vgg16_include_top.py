import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

tf.random.set_seed(333)
np.random.seed(333)
print(tf.__version__)   # 2.7.4

from tensorflow.keras.applications import VGG16
# model = VGG16()
### 디폴트 ###
# model = VGG16(weights='imagenet',
#               include_top=True,
#               input_shape=(224,224,3),
#               )
#############
# model.summary()

###################### VGG-16 기본 모델 ######################
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  input_1 (InputLayer)        [(None, 224, 224, 3)]     0
#  ...
#  predictions (Dense)         (None, 1000)              4097000
# =================================================================
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0
# _________________________________________________________________

model = VGG16(# weights='imagenet',
              include_top=False,
              input_shape=(100,100,3),
              )
model.summary()

#_________________________________________________________________
# Layer (type)                Output Shape              Param #   
#=================================================================
# input_1 (InputLayer)        [(None, 224, 224, 3)]     0

# block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792      

# block1_conv2 (Conv2D)       (None, 224, 224, 64)      36928     

# block1_pool (MaxPooling2D)  (None, 112, 112, 64)      0

# block2_conv1 (Conv2D)       (None, 112, 112, 128)     73856

# block2_conv2 (Conv2D)       (None, 112, 112, 128)     147584

# block2_pool (MaxPooling2D)  (None, 56, 56, 128)       0

# block3_conv1 (Conv2D)       (None, 56, 56, 256)       295168

# block3_conv2 (Conv2D)       (None, 56, 56, 256)       590080

# block3_conv3 (Conv2D)       (None, 56, 56, 256)       590080

# block3_pool (MaxPooling2D)  (None, 28, 28, 256)       0
# block4_conv1 (Conv2D)       (None, 28, 28, 512)       1180160
# block4_conv2 (Conv2D)       (None, 28, 28, 512)       2359808

# block4_conv3 (Conv2D)       (None, 28, 28, 512)       2359808

# block4_pool (MaxPooling2D)  (None, 14, 14, 512)       0

# block5_conv1 (Conv2D)       (None, 14, 14, 512)       2359808

# block5_conv2 (Conv2D)       (None, 14, 14, 512)       2359808

# block5_conv3 (Conv2D)       (None, 14, 14, 512)       2359808

# block5_pool (MaxPooling2D)  (None, 7, 7, 512)         0

#=================================================================
#Total params: 14,714,688
#Trainable params: 14,714,688
#Non-trainable params: 0
#_________________________________________________________________

########################## include_top=False ########################
#1. FC Layer 날려
#2. input_shape를 내가 하고싶은 데이터 shape로 맞춰!!