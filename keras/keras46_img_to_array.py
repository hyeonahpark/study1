from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import numpy as np

path='c:/ai5/_data/image/me/me3.jpg'

img = load_img(path, target_size=(100, 100))
print(img) #<PIL.Image.Image image mode=RGB size=200x200 at 0x1ABCB49A6A0>
print(type(img)) #<class 'PIL.Image.Image'>
plt.imshow(img)
plt.show()

arr = img_to_array(img)
# print(arr)
print(arr.shape)

#차원증가
img = np.expand_dims(arr, axis=0)
np_path = 'c:/ai5/_data/image/me/'
np.save(np_path + 'me3.npy', arr=img)
# print(img.shape)

# me 폴더에 위에 데이터를 npy로 저장할 것