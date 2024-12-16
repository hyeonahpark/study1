import torch

#pytorch  버전 확인
print('Pytorch 버전 : ', torch.__version__)

#cuda 사용 가능 여부
cuda_avilable = torch.cuda.is_available()
print('Cuda 사용 가능 여부 : ', cuda_avilable)

#사용 가능 gpu 갯수 확인
gpu_count = torch.cuda.device_count()
print('사용 사능한 GPU 개수 : ',gpu_count)

if cuda_avilable:
     #현재 사용중인 gpu 장치 확인
     current_device = torch.cuda.current_device()
     print('현재 사용중인 gpu장치 ID : ', current_device)
     print('현재 gpu 이름 :', torch.cuda.get_device_name(current_device))
     
else:
    print('GPU 없다!!!')
    

#CUDA 버전 확인
print('CUDA 버전 :', torch.version.cuda)

#cuDNN qjwjs ghkrdls
cudnn_version = torch.backends.cudnn.version()
if cudnn_version is not None:
     print('cuDNN 버전 :', cudnn_version)
else :
     print('cuDNN 없다!!!') #90100ㅇ