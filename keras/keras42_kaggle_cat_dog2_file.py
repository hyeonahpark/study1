import os
import natsort
import numpy as np

file_path = "C:/ai5/_data/kaggle/dogs-vs-cats-redux-kernels-edition/test/test"
file_names = natsort.natsorted(os.listdir(file_path))

print(np.unique(file_names))
i = 1
for name in file_names:
    src = os.path.join(file_path,name)
    dst = str(i).zfill(5)+ '.jpg'
    dst = os.path.join(file_path, dst)
    os.rename(src, dst)
    i += 1
    
    
    