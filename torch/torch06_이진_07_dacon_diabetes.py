import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, accuracy_score
import pandas as pd

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

#1. data
path = 'C:\\ai5\\_data\\dacon\\diabetes\\' #슬래시 두개는 슬래시 하나로 인식함 (\a 와 \b는 문자열에서 특수문자로 인식하기 때문)
train_csv=pd.read_csv(path + "train.csv", index_col=0)
test_csv=pd.read_csv(path + "test.csv", index_col=0)
sample_submission=pd.read_csv(path + "sample_submission.csv", index_col=0)

print(train_csv.shape)  # (652,9)
print(test_csv.shape)  # (116,8)
print(sample_submission.shape) #(116,1)

################## 결측치 확인 #####################
print(train_csv.isna().sum())
print(train_csv.isnull().sum())
print(test_csv.isna().sum())
print(test_csv.isnull().sum())
###########################################

print(test_csv.info())
print(train_csv.describe())

################# x와 y 분리 ######################
x=train_csv.drop(['Outcome'], axis=1) #대괄호 하나 안에 다 넣기 ! 두개 이상은 리스트
print(x)
print(x.shape) #(652, 8)
y=train_csv['Outcome']
print(y.shape) # (652., )

unique,counts=np.unique(y, return_counts=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True, random_state=369, stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
#print(x_train.shape, y_train.shape) #(521, 8), (521, )

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
model = nn.Sequential(
    nn.Linear(8,16),
    nn.ReLU(),
    nn.Linear(16,32),
    nn.ReLU(),
    nn.Linear(32,32),
    nn.ReLU(),
    nn.Linear(32,64),
    nn.ReLU(),
    nn.Linear(64,64),
    nn.ReLU(),
    nn.Linear(64,32),
    nn.ReLU(),
    nn.Linear(32,16),
    nn.ReLU(),
    nn.Linear(16,8),
    nn.ReLU(),
    nn.Linear(8,1),
    nn.Sigmoid()
).to(DEVICE)

#3. compile
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x, y):
    #model.train() #훈련모드 , 디폴트
    optimizer.zero_grad()
    hypothesis=model(x)
    loss=criterion(hypothesis, y)
    loss.backward()
    optimizer.step()
    
    return loss.item()

epochs = 10000
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
from sklearn.metrics import accuracy_score
y_pre = model(x_test).detach().cpu()
# print(y_pre.shape, y_test.shape) #torch.Size([171, 1]) torch.Size([171, 1])

acc = accuracy_score(y_test.detach().cpu().numpy(), np.round(y_pre.detach().cpu().numpy()))
print('ACC : ', acc)


# 최종 loss :  24.94675636291504
# ACC :  0.7878787878787878