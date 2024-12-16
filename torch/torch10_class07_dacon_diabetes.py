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
path = 'C:\\ai5\\_data\\dacon\\diabetes\\' #슬래시 두개는 슬래시 하나로 인식함 (\a 와 \b는 문자열에서 특수문자로 인식하기 때문)
train_csv=pd.read_csv(path + "train.csv", index_col=0)
test_csv=pd.read_csv(path + "test.csv", index_col=0)
sample_submission=pd.read_csv(path + "sample_submission.csv", index_col=0)

x=train_csv.drop(['Outcome'], axis=1) #대괄호 하나 안에 다 넣기 ! 두개 이상은 리스트
y=train_csv['Outcome']


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=369, stratify=y)

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
        self.linear1 = nn.Linear(input_dim, 16)
        self.linear2 = nn.Linear(16, 32)
        self.linear3 = nn.Linear(32, 32)
        self.linear4 = nn.Linear(32, 16)
        self.linear5 = nn.Linear(16, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
        
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
        x = self.sigmoid(x)
        return x
    
model = Model(8,1).to(DEVICE)
    


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
from sklearn.metrics import accuracy_score
y_pre = model(x_test).detach().cpu()
# print(y_pre.shape, y_test.shape) #torch.Size([171, 1]) torch.Size([171, 1])

acc = accuracy_score(y_test.detach().cpu().numpy(), np.round(y_pre.detach().cpu().numpy()))
print('ACC : ', acc)

# 최종 loss :  32.93278503417969
# ACC :  0.6717557251908397