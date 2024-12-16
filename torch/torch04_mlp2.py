import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용 DEVICE :', DEVICE)

#1. data
x=np.array([[1,2,3,4,5,6,7,8,9,10],
            [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
            [10,9,8,7,6,5,4,3,2,1]]).transpose()
y=np.array([1,2,3,4,5,6,7,8,9,10])
print(x.shape, y.shape) #(10, 3) (10,)

x = torch.FloatTensor(x).to(DEVICE)
y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)

model = nn.Sequential(
    nn.Linear(3,8),
    nn.Linear(8,6),
    nn.Linear(6,4),
    nn.Linear(4,3),
    nn.Linear(3,2),
    nn.Linear(2,1)
).to(DEVICE)

#3.compile

criterion=nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x, y):
    optimizer.zero_grad()
    hypothesis=model(x)
    loss=criterion(hypothesis, y)
    loss.backward()
    optimizer.step()
    
    return loss.item()

epochs = 3000

for epoch in range(1, epochs+1):
    loss=train(model, criterion, optimizer, x, y)
    print('epochs : {}, loss : {}'.format(epoch, loss))
    

#4. predict
def evaluate(model, criterion, x, y):
    model.eval()
    
    with torch.no_grad():
        y_predict = model(x)
        loss2 = criterion(y, y_predict)
    return loss2.item()

loss2=evaluate(model, criterion, x, y)
print('최종 loss :', loss2)

results = model(torch.Tensor([[10,1.3,1]]).to(DEVICE))
print('10, 1.3, 1의 예측값 :', results.item())

# 최종 loss : 1.0570033563694903e-11
# 10, 1.3, 1의 예측값 : tensor([[10.0000]])