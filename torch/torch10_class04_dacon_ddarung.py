import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, accuracy_score
import pandas as pd

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

#1. DATA
path = "./_data/dacon/따릉이/"

train_csv = pd.read_csv(path + "train.csv", index_col=0) #. 하나는 root 라는 뜻, 그 하단은 /로 표현, index_col=0을 해주면 0번째인 id가 인덱스라는 것을 표현함
print(train_csv) # [1459 rows x 10 columns]

test_csv = pd.read_csv(path + "test.csv", index_col=0) #. 하나는 root 라는 뜻, 그 하단은 /로 표현, index_col=0을 해주면 0번째인 id가 인덱스라는 것을 표현함
print(test_csv) # [715 rows x 9 columns]

submission_csv = pd.read_csv(path + "submission.csv", index_col=0) #. 하나는 root 라는 뜻, 그 하단은 /로 표현, index_col=0을 해주면 0번째인 id가 인덱스라는 것을 표현함
print(submission_csv) # [715 rows x 1 columns]
train_csv=train_csv.dropna() #결측치 포함 행 제거
test_csv = test_csv.fillna(test_csv.mean()) #fillna 함수 : 결측치를 채운다
x = train_csv.drop(['count'], axis=1) #train_csv에서 count 열 삭제 후 x에 넣기
y = train_csv['count'] #train_csv에서 count 열만 y에 넣기

print(x.shape, y.shape) #(1328, 9) (1328,)

#print(type(x), type(y)) #<class 'pandas.core.frame.DataFrame'> <class 'pandas.core.series.Series'>

x = x.to_numpy()
y = y.to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=4343)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
#x_train = torch.DoubleTensor(x_train).to(DEVICE)

x_test = torch.FloatTensor(x_test).to(DEVICE)
#x_test = torch.DoubleTensor(x_test).to(DEVICE)

y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
#y_train = torch.LongTensor(y_train).unsqueeze(1).to(DEVICE)
# y_train = torch.IntTensor(y_train).unsqueeze(1).to(DEVICE)

y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)
#y_test = torch.LongTensor(y_test).unsqueeze(1).to(DEVICE)
# y_test = torch.IntTensor(y_test).unsqueeze(1).to(DEVICE)

# print("=============================")
# print(x_train.shape, x_test.shape) #torch.Size([569, 30]) torch.Size([569, 30])
# print(y_train.shape, y_test.shape) #torch.Size([398, 1]) torch.Size([171, 1])
# print(type(x_train), type(y_train)) #<class 'torch.Tensor'> <class 'torch.Tensor'>

#2.modeling
# model = nn.Sequential(
#     nn.Linear(30,64),
#     nn.ReLU(),
#     nn.Linear(64,32),
#     nn.ReLU(),
#     nn.Linear(32,32),
#     nn.ReLU(),
#     nn.Linear(32,16),
#     nn.Linear(16,1),
#     nn.Sigmoid()
# ).to(DEVICE)

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        #super().__init__() #default
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 32)
        self.linear4 = nn.Linear(32, 16)
        self.linear5 = nn.Linear(16, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    ## 순전파 !!!
    def forward(self, input_size): # method
        x = self.linear1(input_size)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.linear5(x)
        return x
    
model = Model(9,1).to(DEVICE)
    


#3. compile
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x, y):
    #model.train() #훈련모드 , 디폴트
    optimizer.zero_grad()
    hypothesis=model(x)
    loss=criterion(hypothesis, y)
    loss.backward()
    optimizer.step()
    
    return loss.item()

epochs = 1000
for epoch in range(1, epochs+1):
    loss=train(model, criterion, optimizer, x_train, y_train)
    print('epochs : {}, loss : {}'.format(epoch, loss)) #verbose

print('===================================================')

#4. predict
#loss = model.evaluate(x,y)
def evaluate(model, criterion, x, y): #가중치 갱신이 필요없어서 optimizer x
    model.eval() # 평가모드, 기울기계산, 가중치 갱신을 하지 말아라하는 의미, 드랍아웃, batch normalization도 적용 x
    
    with torch.no_grad():
        y_predict = model(x)
        loss2 = criterion(y, y_predict) #loss 최종값
    return loss2.item()

last_loss=evaluate(model, criterion, x_test, y_test)
print('최종 loss : ', last_loss)

############밑에 완성하기##################
from sklearn.metrics import r2_score
y_pre = model(x_test).detach().cpu()
# print(y_pre.shape, y_test.shape) #torch.Size([171, 1]) torch.Size([171, 1])

acc = r2_score(y_test.detach().cpu().numpy(), np.round(y_pre.detach().cpu().numpy()))
print('ACC : ', acc)

# 최종 loss :  2177.900390625
# ACC :  0.6374717950820923