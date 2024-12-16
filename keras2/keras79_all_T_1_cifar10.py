"""
01. VGG19
02. Xception
03. ResNet50
04. ResNet101
05. InceptionV3
06. InceptionResNetV2
07. DenseNet121
08. MobileNetV2
09. NasNetMobile
10. EfficeientNetB0

#GAP 서서 기존거와 최고 성능 비교

#keras 76_GAP3_cifar100
#keras 76_GAP4_horse
#keras 76_GAP5_rps
#keras 76_GAP6_kaggle_cat_dog
#keras 76_GAP7_men_women
"""

import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import (
    VGG16, VGG19, Xception, ResNet50, ResNet101,
    InceptionV3, InceptionResNetV2, DenseNet121,
    MobileNetV2, NASNetMobile, EfficientNetB0
)
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import EarlyStopping

# 1. 데이터 로드 및 정규화
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

# 데이터 전처리: tf.data로 리사이즈 및 배치 처리
def preprocess(image, label):
    image = tf.image.resize(image, (128, 128))  # 크기를 128x128로 설정
    return image, label

BATCH_SIZE = 8
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = (
    train_dataset.map(preprocess, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = (
    test_dataset.map(preprocess, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

# 2. NASNetMobile 모델 정의
base_model = NASNetMobile(include_top=False, input_shape=(128, 128, 3))  # include_top=False로 설정
base_model.trainable = False  # 가중치 동결

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(100, activation='relu'),
    Dense(100, activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# 3. 콜백 설정
es = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

# 4. 모델 학습
print("\nStarting model training with NASNetMobile...")
hist = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=50,
    callbacks=[es],
    verbose=1
)

# 5. 모델 평가
print("\nEvaluating the model...")
loss, acc = model.evaluate(test_dataset)
print(f"\nNASNetMobile - Loss: {loss:.4f}, Accuracy: {acc:.4f}")




#flatten
# 동결x
# loss :  0.6866620182991028
# ACC :  0.795
# 걸린 시간 :  151.32 초


# GAP
# loss :  2.303164005279541
# ACC :  0.1
# ACC score : 1.0


# 모델: VGG16, Loss: 1.1375, Accuracy: 0.6045, Time: 73.0 초
# 모델: VGG19, Loss: 1.1805, Accuracy: 0.5892, Time: 87.58 초
# 모델: Xception, Loss: 0.8369, Accuracy: 0.0929
# 모델: ResNet50, Loss: 1.6756, Accuracy: 0.4061, Time: 92.55 초
# 모델: ResNet101, Loss: 1.8207, Accuracy: 0.3512, Time: 540.71 초
# 모델: InceptionV3, Loss: 1.1214, Accuracy: 0.1023, Time: 351.38 초
# 모델: InceptionResNetV2, Loss: 0.8635, Accuracy: 0.0888, Time: 741.5 초
# 모델: DenseNet121, Loss: 1.0431, Accuracy: ,0.6388 Time: 99.15 초
# 모델: MobileNetV2, Loss: 1.8039, Accuracy: 0.3537, Time: 100.02 초
