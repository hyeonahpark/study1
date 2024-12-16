import numpy as np
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16, ResNet50, ResNet101, DenseNet121, MobileNetV2, EfficientNetB0
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 모델 리스트
model_list = [
    VGG16(include_top=False, input_shape=(224, 224, 3)),
    ResNet50(include_top=False, input_shape=(224, 224, 3)),
    ResNet101(include_top=False, input_shape=(224, 224, 3)),
    DenseNet121(include_top=False, input_shape=(224, 224, 3)),
    MobileNetV2(include_top=False, input_shape=(224, 224, 3)),
    EfficientNetB0(include_top=False, input_shape=(224, 224, 3)),
]

# 데이터 로드 및 전처리
train_datagen = ImageDataGenerator(rescale=1./255)
path = './_data/image/rps/'

xy_train = train_datagen.flow_from_directory(
    path, 
    target_size=(224, 224),  # 모델 요구 사항에 맞는 크기
    batch_size=1100,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True
)

# 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(xy_train[0][0], xy_train[0][1], train_size=0.9, random_state=5656)

for model_i in model_list:
    # 모델 초기화 및 동결 설정
    model_i.trainable = False

    model = Sequential([
        model_i,
        GlobalAveragePooling2D(),
        Dense(3, activation='sparse')
    ])

    # 모델 컴파일
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 모델 학습
    start = time.time()
    hist = model.fit(
        x_train, y_train,
        validation_split=0.2,
        epochs=1000,  # 테스트용으로 10회로 제한
        batch_size=32,
        verbose=0
    )
    end = time.time()
    
    from tensorflow.keras.callbacks import EarlyStopping
    es = EarlyStopping(monitor='val_loss', mode='min', 
                    patience=10, verbose=0,
                    restore_best_weights=True,
                    )

    start = time.time()
    hist = model.fit(x_train, y_train, epochs=1000, batch_size=128,
            verbose=0, 
            validation_split=0.2,
            callbacks=[es],
            )
    end = time.time()

    #4. 평가, 예측
    loss = model.evaluate(x_test, y_test, verbose=1)
    # print('loss :', loss[0])
    # print('acc :', round(loss[1],2))
    
    print("-----------------------")
    print("모델명 :", model_i.name, 'loss :', loss[0], 'acc :', round(loss[1],2))
    
    y_pre = model.predict(x_test)

    y_pre = np.argmax(y_pre, axis=1).reshape(-1,1)
    y_test = np.argmax(y_test, axis=1).reshape(-1,1)

    r2 = accuracy_score(y_test, y_pre)


# 모델명 : vgg16 loss : 5.7283041314803995e-06 acc : 1.0