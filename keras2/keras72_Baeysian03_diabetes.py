import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import r2_score

from bayes_opt import BayesianOptimization
import time
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. 데이터
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=336, train_size=0.8)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

print(x_train.shape, y_train.shape)  # (353, 10) (353,)

# LabelEncoder 설정
le = LabelEncoder()
label_encoder = LabelEncoder()

# 2. 모델 정의
def build_model(drop=0.5, optimizer='adam', activation='relu',
                node1=128, node2=64, node3=32, node4=16, node5=8, lr=0.001):
    optimizer = le.inverse_transform([int(optimizer)])[0]  # 정수형 인덱스를 문자열로 변환
    activation = label_encoder.inverse_transform([int(activation)])[0]  # 정수형 인덱스를 문자열로 변환
    
    inputs = Input(shape=(10,), name='inputs')
    x = Dense(int(node1), activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(int(node2), activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(int(node3), activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    x = Dense(int(node4), activation=activation, name='hidden4')(x)
    x = Dense(int(node5), activation=activation, name='hidden5')(x)
    outputs = Dense(1, activation='linear', name='outputs')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['mae'], loss='mse')
    
    model.fit(x_train, y_train, epochs=100, validation_split=0.1, verbose=0)
    
    y_pred = model.predict(x_test)
    result = r2_score(y_test, y_pred)
    
    return result

# 3. 하이퍼파라미터 생성 함수
def create_hyperparameter():
    optimizers = ['adam', 'rmsprop', 'adadelta']
    optimizers = (0, len(optimizers) - 1)  # LabelEncoder 변환 범위 설정
    dropouts = (0.2, 0.5)
    activations = ['relu', 'elu', 'selu', 'linear']
    activations = (0, len(activations) - 1)  # LabelEncoder 변환 범위 설정
    node1 = [16, 128]
    node2 = [16, 128]
    node3 = [16, 128]
    node4 = [16, 128]
    node5 = [16, 128]
    
    return {
        'optimizer': optimizers,
        'drop': dropouts,
        'activation': activations,
        'node1': (16, 128),  # 정수형 범위
        'node2': (16, 128),
        'node3': (16, 128),
        'node4': (16, 128),
        'node5': (16, 128),
    }

# 4. Bayesian Optimization 설정
hyperparameters = create_hyperparameter()

bay = BayesianOptimization(
    f=lambda drop, optimizer, activation, node1, node2, node3, node4, node5: 
      build_model(drop, int(optimizer), int(activation), int(node1), int(node2), int(node3), int(node4), int(node5)),
    pbounds=hyperparameters,
    random_state=333
)

# 5. 최적화 과정
n_iter = 100
start_time = time.time()
bay.maximize(init_points=5, n_iter=n_iter)
end_time = time.time()

# 6. 결과 출력
print(bay.max)
print(n_iter, '번 걸린 시간 :', round(end_time - start_time, 2), '초')
