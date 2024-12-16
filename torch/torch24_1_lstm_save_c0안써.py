import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

random.seed(333)
np.random.seed(333)
torch.manual_seed(333)  # torch 고정, cpu 고정 
torch.cuda.manual_seed(333) # gpu 고정

# USE_CUDA = torch.cuda.is_available()
# DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
# print('torch :', torch.__version__, '사용 DEVICE :', DEVICE)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

import pandas as pd
import numpy as np

path = 'C:/ai5/_data/kaggle/netflix/'
train_csv = pd.read_csv(path + "train.csv")
print(train_csv)
print(train_csv.info())
print(train_csv.describe())


#pandas 는 항상 열먼저 나옴
import matplotlib.pyplot as plt
# data = train_csv.iloc[:, 1:4].values #index location, iloc는 무조건 숫자형태,
# data = (data-np.min(data))/ (np.max(data)-np.min(data))
# data = pd.DataFrame(data)
# #data = train_csv[1:4]
# data['증가'] = train_csv['Close']
# data = train_csv.iloc[:, 1:4].values #index location, iloc는 무조건 숫자형태,
# data = (data-np.min(data, axis=0))/ (np.max(data, axis=0)-np.min(data, axis=0)) #컬럼별로 scaling 해야함
# data = pd.DataFrame(data)
# print(data.describe())
# hist = data.hist()
# plt.show()


from torch.utils.data.dataset import Dataset, TensorDataset
from torch.utils.data import DataLoader

class Custom_Dataset(Dataset):
    def __init__(self):
        self.csv = train_csv
        self.x = self.csv.iloc[:, 1:4].values  #.values = numpy
        self.x = (self.x-np.min(self.x, axis=0))/ (np.max(self.x, axis=0)-np.min(self.x, axis=0))

        #정규화
        self.y = self.csv['Close'].values
        self.y = (self.y-np.min(self.y, axis=0))/ (np.max(self.y, axis=0)-np.min(self.y, axis=0))

        
    def __len__(self): # tensor 데이터 형태로 만들기 위해서 길이가 얼마나 필요한지
        return len(self.x)-30 
    
    def __getitem__(self, i):
        x = self.x[i:i+30] # 시계열 데이터 형태
        y = self.y[i+30]
    
        return x, y
    
aaa = Custom_Dataset()
print(aaa) #<__main__.Custom_Dataset object at 0x000001DE2DCE0B90>
print(type(aaa)) #<class '__main__.Custom_Dataset'>

print(aaa[0])
print(aaa[0][0].shape) #(30, 3)
print(aaa[0][1]) #94
print(len(aaa)) #937
# print(aaa[937]) #936번째까지 있지. 936번째놈이 끝!

#### x는 (937, 30, 3), y는 (937, 1)

train_loader = DataLoader(aaa, batch_size=32)

aaa = iter(train_loader)
bbb = next(aaa)
print(bbb)
print(bbb[0].size()) #torch.Size([32, 30, 3])


#2. modeling

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        
        self.cell = nn.LSTM(input_size=3,
                          hidden_size=64,
                          num_layers=5,
                          batch_first=True,
                          bidirectional=False
                          )
        self.fc1 = nn.Linear(in_features=30*64, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=1)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):        
        x, _ = self.cell(x)
        x = self.relu(x)
        x = x.contiguous()    
        x = x.view(-1, 30*64)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x        

model = LSTM().to(DEVICE)

#3. compile

from torch.optim import Adam
optim = Adam(params= model.parameters(), lr=0.001)

import tqdm #iterator를 tqdm 으로 감싸야함

for epoch in range(1, 281):
    iterator = tqdm.tqdm(train_loader)
    for x,y in iterator:
        optim.zero_grad()
        
        #h0 = torch.zeros(5, x.shape[0], 64).to(DEVICE) #(num_layers, batch_size, hidden_size) = (5, 12, 64)
        #c0 = torch.zeros(5, x.shape[0], 64).to(DEVICE)
        hypothesis = model(x.type(torch.FloatTensor).to(DEVICE))
        
        loss = nn.MSELoss()(hypothesis, y.type(torch.FloatTensor).to(DEVICE))
        loss.backward()
        
        optim.step()
        
        iterator.set_description(f'epoch:{epoch} loss:{loss.item()}')
        
save_path = 'C:\\ai5\\_save\\torch\\'
torch.save(model.state_dict(), save_path+'t24_1_y_scaling.pth')

"""
save_path = 'C:\\ai5\\_save\\torch\\'

# 4. predict

# `Custom_Dataset()` 객체로 다시 DataLoader 설정
train_loader = DataLoader(Custom_Dataset(), batch_size=1)
y_predict = []
y_true=[]
total_loss = 0

with torch.no_grad():
    model.load_state_dict(torch.load(save_path + 't23_1_y_scaling.pth', map_location=DEVICE))
    for x_test, y_test in train_loader:
        h0 = torch.zeros(5, x_test.shape[0], 64).to(DEVICE)  # (num_layers, batch_size, hidden_size) = (5, 1, 64)
        c0 = torch.zeros(5, x_test.shape[0], 64).to(DEVICE)
        y_pred = model(x_test.type(torch.FloatTensor).to(DEVICE), h0, c0)
        y_predict.append(y_pred.cpu().numpy())  # numpy 배열로 변환 후 저장
        y_true.append(y_test.cpu().numpy())  # numpy 배열로 변환 후 저장
        
        loss = nn.MSELoss()(y_pred, y_test.type(torch.FloatTensor).to(DEVICE))
        total_loss += loss / len(train_loader)

#print(f'y_predict : {y_predict}, \n shape: {y_predict.shape}')

## 실습 ##
# R2 맹글기
# R^2 계산
from sklearn.metrics import r2_score

# numpy 배열로 변환하여 r2 계산
y_predict = np.array(y_predict).flatten()
y_true = np.array(y_true).flatten()

r2 = r2_score(y_true, y_predict)
print('R2:', r2) # R2: 0.9737935276602581
print('total_loss:', total_loss.item()) #total_loss: 0.00250460603274405
"""