import tensorflow as tf

print('텐서플로 버전 : ', tf.__version__) # 2.7.4

if tf.config.list_physical_devices('GPU') :
    print('GPU 있다!!')

else:
    print('GPU 없다!!')
    
    
#CUDA 버전
cuda_version = tf.sysconfig.get_build_info()['cuda_version']
print('CUDA 버전 :', cuda_version) #CUDA 버전 : 64_112

#cuDNN 버전
cudnn_version = tf.sysconfig.get_build_info()['cudnn_version']
print('cuDNN 버전 :', cudnn_version) #cuDNN 버전 : 64_8
