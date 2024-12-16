import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

# 1. DATA
datasets = load_digits()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=0)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.LongTensor(y_train).to(DEVICE)  # y는 LongTensor로 유지
y_test = torch.LongTensor(y_test).to(DEVICE)    # y는 LongTensor로 유지

# 1. 데이터셋 만들기
train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)

# 2. 데이터 로더 만들기
train_loader = DataLoader(train_set, batch_size=512, shuffle=True)
test_loader = DataLoader(test_set, batch_size=512, shuffle=False)

# 3. 모델 정의
class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 32)
        self.linear4 = nn.Linear(32, 16)
        self.linear5 = nn.Linear(16, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, input_size):
        x = self.linear1(input_size)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.linear5(x)
        return x  # 여기서는 FloatTensor로 출력됨

model = Model(64, 10).to(DEVICE)  # 출력 차원(output_dim) = 10 (다중 클래스)

# 4. 컴파일
criterion = nn.CrossEntropyLoss()  # 다중 클래스 분류 손실 함수
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, loader):
    total_loss = 0
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        hypothesis = model(x_batch)  # 모델 출력은 FloatTensor
        loss = criterion(hypothesis, y_batch)  # y_batch는 LongTensor
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# 5. 훈련
epochs = 100
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, train_loader)
    print(f'epochs: {epoch}, loss: {loss}')

print('===================================================')

# 6. 정확도 계산 (Dataloader 사용)
model.eval()  # 평가 모드
y_true = []
y_pred = []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        y_pre = torch.argmax(model(x_batch), dim=1).detach().cpu()  # argmax로 클래스 예측
        y_true.extend(y_batch.cpu().numpy())  # 실제 값 저장
        y_pred.extend(y_pre.numpy())  # 예측 값 저장

# Accuracy 계산
acc = accuracy_score(y_true, y_pred)
print('ACC : ', acc)

# 7. 평가 함수 정의
def evaluate(model, criterion, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            y_predict = model(x_batch)
            loss = criterion(y_predict, y_batch)
            total_loss += loss.item()
    return total_loss / len(loader)

last_loss = evaluate(model, criterion, test_loader)
print('최종 loss : ', last_loss)

#ACC :  0.9805555555555555
#최종 loss :  0.4611395597457886