import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, accuracy_score


USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

#1. data
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

# x = torch.FloatTensor(x)
# y = torch.LongTensor(y)
print(x.shape, y.shape) #torch.Size([581012, 54]) torch.Size([581012])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, shuffle = True, random_state=1186, stratify=y
    )

# print(x_train.size(), y_train.size()) #torch.Size([112, 4]) torch.Size([112])
# print(x_test.size(), y_test.size()) #torch.Size([38, 4]) torch.Size([38]) 

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

y_train = y_train - 1
y_test = y_test - 1

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.LongTensor(y_train).to(DEVICE)
y_test = torch.LongTensor(y_test).to(DEVICE)

#2. model
model = nn.Sequential(
    nn.Linear(54, 100),
    nn.ReLU(),
    nn.Linear(100, 200),
    nn.ReLU(),
    nn.Linear(200, 300),
    nn.ReLU(),
    nn.Linear(300, 400),
    nn.ReLU(),
    nn.Linear(400, 300),
    nn.ReLU(),
    nn.Linear(300, 200),
    nn.ReLU(),
    nn.Linear(200, 100),
    nn.ReLU(),
    nn.Linear(100, 7)
).to(DEVICE)

#3. compile
criterion = nn.CrossEntropyLoss() #sparse categorical entropy

optimizer = optim.Adam(model.parameters(), lr = 0.01)

def train(model, criterion, optimizer, x_train, y_train):
    # model.train()
    optimizer.zero_grad() #기울기 연산이 누적되기 때문에 0으로 초기화
    hypothesis=model(x_train)
    loss=criterion(hypothesis, y_train)
    loss.backward()
    optimizer.step()
    
    return loss.item()


EPOCHS = 1000
for epoch in range(1, EPOCHS+1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print('epoch : {}, loss : {:.8f}'.format(epoch, loss))
    
#4. predict
def evaluate(model, criterion, x_test, y_test):
    # model.eval() #가중치 갱신x, 기울기 계산 할 수도 있땅
    with torch.no_grad():
        hypothesis = model(x_test)
        loss = criterion(hypothesis, y_test) #loss 최종값
    return loss.item()

loss=evaluate(model, criterion, x_test, y_test)
print('최종 loss : ', loss)

########## acc 출력 ##########
from sklearn.metrics import accuracy_score
y_pre = model(x_test).detach().cpu()
y_predict = torch.argmax(model(x_test), dim=1)
# print(y_pre.shape, y_test.shape) #torch.Size([171, 1]) torch.Size([171, 1])

acc = accuracy_score(y_test.detach().cpu().numpy(), np.argmax(y_pre.detach().cpu().numpy(), axis=1))
print('ACC : ', acc)

score = (y_predict == y_test).float().mean()
print(f'accuracy : {score:.4f}')

# 최종 loss :  0.29994815587997437
# ACC :  0.8796599084368869
# accuracy : 0.8797