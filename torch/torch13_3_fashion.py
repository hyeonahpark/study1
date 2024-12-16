import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import FashionMNIST

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch:', torch.__version__, '사용 DEVICE : ', DEVICE)

path = './study/torch/_data/'
train_dataset = FashionMNIST(path, train = True, download= True) #다운로드 (토치는 다운로드 받아줘야함! 케라스는 안받아도 됨)
test_dataset = FashionMNIST(path, train = False, download= True)  #다운로드 안됨.

print(train_dataset)
print(type(train_dataset))
print(train_dataset[0])
print(train_dataset[0][0])

x_train, y_train = train_dataset.data/255., train_dataset.targets
x_test, y_test = test_dataset.data/255., test_dataset.targets

print(x_train)
print(y_train)
print(x_train.shape, y_train.size()) #torch.Size([60000, 28, 28]) torch.Size([60000])
print(np.min(x_train.numpy()), np.max(x_train.numpy()))

x_train, x_test = x_train.view(-1, 28*28), x_test.reshape(-1, 784)
print(x_train.shape, x_test.size()) #torch.Size([60000, 784]) torch.Size([10000, 784])


train_dset = TensorDataset(x_train, y_train)
test_dset = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dset, batch_size = 32, shuffle=True)
test_loader = DataLoader(test_dset, batch_size = 32, shuffle=False)

#2. model

class DNN(nn.Module) : #클래스 괄호 안에는 상속
    def __init__(self, num_features):
        super().__init__()
        # self(self, DNN).__init__() #이렇게 써도 같음
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.hidden_layer3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.hidden_layer4 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.hidden_layer5 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.output_layer = nn.Linear(32,10)
        
    def forward(self, x) :
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = self.hidden_layer4(x)
        x = self.hidden_layer5(x)
        x = self.output_layer(x)
        return x
        
model = DNN(784).to(DEVICE)

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

