import autokeras as ak
import tensorflow as tf
print(ak.__version__) #1.0.15
print(tf.__version__) #2.7.4
import time

#1. data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(x_train.shape, x_test.shape)

#2. modeling
model = ak.ImageClassifier(
    overwrite=False,
    max_trials=3,
)

#3. compile
start_time = time.time()
model.fit(x_train, y_train, epochs=10, validation_split=0.15)
end_time = time.time()


###### 최적의 출력 모델 #######
best_model = model.export_model()
print(best_model.summary())

##### 최적의 모델 저장 #######
path = 'C:\\ai5\\_save\\autokeras\\'
best_model.save(path + 'keras70_autokeras1.h5')

#4. predict
y_predict = model.predict(x_test)
results = model.evaluate(x_test, y_test)
print('model 결과 : ', results)

y_predict2 = best_model.predict(x_test)
# results2 = best_model.evaluate(x_test, y_test)
# print('best_model 결과 :', results2)

#model 결과 :  [0.03567328676581383, 0.9890000224113464]