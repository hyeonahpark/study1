import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import CIFAR100

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch:', torch.__version__, '사용 DEVICE :', DEVICE)

path = './study/torch/_data'
train_dataset = CIFAR100(path, train=True, download=True)
test_dataset = CIFAR100(path, train=False, download=True)

# 데이터 정규화 및 PyTorch 텐서로 변환
x_train, y_train = train_dataset.data / 255.0, train_dataset.targets
x_test, y_test = test_dataset.data / 255.0, test_dataset.targets

x_train, x_test = x_train.reshape(-1, 32*32*3), x_test.reshape(-1, 32*32*3)
print(x_train.shape, len(y_train)) #(50000, 3072) (50000, )

# NumPy 배열을 PyTorch 텐서로 변환
x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.LongTensor(y_train).to(DEVICE)
y_test = torch.LongTensor(y_test).to(DEVICE)

# TensorDataset 생성
train_dset = TensorDataset(x_train, y_train)
test_dset = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dset, batch_size=32, shuffle=False)

#2. 모델 정의
class DNN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(num_features, 100),
            nn.ReLU()
        )
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(100, 200),
            nn.ReLU()
        )
        self.hidden_layer3 = nn.Sequential(
            nn.Linear(200, 300),
            nn.ReLU()
        )
        self.hidden_layer4 = nn.Sequential(
            nn.Linear(300, 400),
            nn.ReLU()
        )
        self.hidden_layer5 = nn.Sequential(
            nn.Linear(400, 500),
            nn.ReLU()
        )
        self.hidden_layer6 = nn.Sequential(
            nn.Linear(500, 400),
            nn.ReLU()
        )
        self.hidden_layer7 = nn.Sequential(
            nn.Linear(400, 300),
            nn.ReLU()
        )
        self.hidden_layer8 = nn.Sequential(
            nn.Linear(300, 200),
            nn.ReLU()
        )
        self.hidden_layer9 = nn.Sequential(
            nn.Linear(200, 100),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(100, 100)  # CIFAR10은 10개의 클래스를 예측하므로 output 크기는 10
        
    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = self.hidden_layer4(x)
        x = self.hidden_layer5(x)
        x = self.hidden_layer6(x)
        x = self.hidden_layer7(x)
        x = self.hidden_layer8(x)
        x = self.hidden_layer9(x)  
        x = self.output_layer(x)
        return x

model = DNN(3072).to(DEVICE)

#3. 컴파일
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def train(model, criterion, optimizer, loader):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        
        optimizer.zero_grad()
        hypothesis = model(x_batch)
        loss = criterion(hypothesis, y_batch)
        loss.backward()
        optimizer.step()
        
        y_predict = torch.argmax(hypothesis, 1)
        acc = (y_predict == y_batch).float().mean()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    
    return epoch_loss / len(loader), epoch_acc / len(loader)

def evaluate(model, criterion, loader):
    model.eval()
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

EPOCH = 200
for epoch in range(1, EPOCH + 1):
    loss, acc = train(model, criterion, optimizer, train_loader)
    val_loss, val_acc = evaluate(model, criterion, test_loader)
    print(f'epoch: {epoch}, loss : {loss:.4f}, acc:{acc:.3f}, val_loss :{val_loss:.4f}, val_acc:{val_acc:.3f}')
    
#4. 마지막 평가
last_loss, acc = evaluate(model, criterion, test_loader)
print('최종 loss : ', last_loss)
print('acc : ', acc)

# 최종 loss :  3.7061649420010014
# acc :  0.1295926517571885

#최종 loss :  3.279639765858269
#acc :  0.22513977635782748