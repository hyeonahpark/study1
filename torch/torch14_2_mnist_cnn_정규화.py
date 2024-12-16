import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import MNIST

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch:', torch.__version__, '사용 DEVICE : ', DEVICE)


import torchvision.transforms as tr
transf = tr.Compose([tr.Resize(56), tr.ToTensor(), tr.Normalize((0.5, ), (0.5))]) # minmax(X) - 평균(0.5)고정 / 표준편차(0.5)고정

#1. data
path = './study/torch/_data/'
train_dataset = MNIST(path, train = True, download= False, transform=transf) #다운로드 (토치는 다운로드 받아줘야함! 케라스는 안받아도 됨)
test_dataset = MNIST(path, train = False, download= False, transform=transf)  #다운로드 안됨.

# print(train_dataset[0][0].shape) #torch.Size([1, 150, 150])
# print(train_dataset[0][1]) #5

#### 정규화 #### /255.
# x_train, y_train = train_dataset.data/255., train_dataset.targets
# x_test, y_test = test_dataset.data/255., test_dataset.targets


### x_train/127.5-1 범위는? -1~1 #정규화라기보다는 표준화와 가까움 (z - 정규화)


train_loader = DataLoader(train_dataset, batch_size = 32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size = 32, shuffle=False)

#2. model

class CNN(nn.Module) : #클래스 괄호 안에는 상속
    def __init__(self, num_features):
        super().__init__()
        # self(self, DNN).__init__() #이렇게 써도 같음
        self.hidden_layer1 = nn.Sequential(
            nn.Conv2d(num_features, 64, kernel_size=(3,3), stride=1), #(n, 64, 54, 54)
            # model.Conv2D(64, (3,3), stride=1, input_shape=(56,56,1))
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)), #(n, 64, 27, 27)
            nn.Dropout(0.2)
        ) 
        
        self.hidden_layer2 = nn.Sequential(
            nn.Conv2d(64, 32,  kernel_size=(3,3), stride=1), #(n, 32, 25, 25)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)), #(n, 32, 12, 12)
            nn.Dropout(0.2)
        )
        
        self.hidden_layer3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=(3,3), stride=1), #(n, 16, 10, 10)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)), #(n, 16, 5, 5)
            nn.Dropout(0.2)
        )
        
        self.hidden_layer4 = nn.Linear(16*5*5, 16)
        self.output_layer = nn.Linear(in_features=16, out_features=10)
        
    def forward(self, x) :
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = x.view(x.shape[0], -1)
        #x = flatten()(x) 케라스에서는 이렇게 씀
        x = self.hidden_layer4(x)
        x = self.output_layer(x)
        return x
        
model = CNN(1).to(DEVICE)

#3. compile
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-4)

def train(model, criterion, optimizer, loader):
    # model.train()
    epoch_loss = 0
    epoch_acc = 0
    
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        
        optimizer.zero_grad()
        
        hypothesis = model(x_batch)     # y = xw + b
        loss = criterion(hypothesis, y_batch)
        
        loss.backward()
        optimizer.step()

        y_predict = torch.argmax(hypothesis, 1)
        acc = (y_predict == y_batch).float().mean()     # y_predict == y_batch : True, False으로 결과 나옴

        epoch_loss += loss.item() 
        epoch_acc += acc
    return epoch_loss / len(loader), epoch_acc / len(loader)


def evaluate(model, criterion, loader):
    #model.eval()
    
    epoch_loss = 0
    epoch_acc = 0
    
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            hypothesis = model(x_batch)
            loss = criterion(hypothesis, y_batch)
            
            y_predict = torch.argmax(hypothesis, 1)
            acc = (y_predict == y_batch).float().mean()
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(loader), epoch_acc / len(loader)
    

EPOCH = 100
for epoch in range(1, EPOCH+1):
    loss, acc = train(model, criterion, optimizer, train_loader)
    val_loss, val_acc = evaluate(model, criterion, test_loader)
    print(f'epoch: {epoch}, loss : {loss:.4f}, acc:{acc:.3f}, val_loss :{val_loss:.4f}, val_acc:{val_acc:.3f}')
    
#epoch: 20, loss : 0.0500, acc:0.985, val_loss :0.1189, val_acc:0.967 

    
#4. predict
last_loss, acc = evaluate(model, criterion, test_loader)
print('최종 loss : ', last_loss)
print('acc : ', acc)

# 최종 loss :  0.03624549519906596
# acc :  0.9888178913738019