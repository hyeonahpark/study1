import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, accuracy_score
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

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


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=0)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)

# 1. 데이터셋 만들기
train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)

# 2. 데이터 로더 만들기
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# 3. 모델 정의
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
    

# 4. 컴파일
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, loader):
    total_loss = 0
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        hypothesis = model(x_batch)
        loss = criterion(hypothesis, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# 5. 훈련
epochs = 1000
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
        y_pre = model(x_batch).detach().cpu()
        y_true.extend(y_batch.cpu().numpy())  # 실제 값 저장
        y_pred.extend(np.round(y_pre.numpy()))  # 예측 값 저장 (0.5 이상을 1로 변환)

# Accuracy 계산
acc = r2_score(y_true, y_pred)
print('ACC : ', acc)

# 7. 평가 함수 정의
def evaluate(model, criterion, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            y_predict = model(x_batch)
            loss = criterion(y_batch, y_predict)
            total_loss += loss.item()
    return total_loss / len(loader)

last_loss = evaluate(model, criterion, test_loader)
print('최종 loss : ', last_loss)

#ACC :  0.7061621132783458
#최종 loss :  2106.8915771484376