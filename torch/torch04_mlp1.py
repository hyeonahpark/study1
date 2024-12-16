import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용 DEVICE :', DEVICE)

#1. data
x=np.array([[1,2,3,4,5,6,7,8,9,10],
            [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3]]).transpose()
y=np.array([1,2,3,4,5,6,7,8,9,10])

print(x.shape, y.shape) #(10, 2) (5,)
x = torch.FloatTensor(x).to(DEVICE)
# print(x)
y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)
### 맹그러바 예측 [10, 1.3, 1]
model = nn.Sequential(
    nn.Linear(2,4),
    nn.Linear(4,3),
    nn.Linear(3,2),
    nn.Linear(2,1),
    # nn.Linear(5,4),
    # nn.Linear(4,3),
    # nn.Linear(3,2),
    # nn.Linear(2,1)
).to(DEVICE)


#3.compile
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x, y):
    optimizer.zero_grad()
    hypothesis=model(x)
    loss=criterion(hypothesis, y)
    loss.backward()
    optimizer.step()
    
    return loss.item()


epochs = 5000

for epoch in range(1, epochs+1):
    loss=train(model, criterion, optimizer, x, y)
    print('epochs : {}, loss : {}'.format(epoch, loss))

print('===================================================')

#4. predict
#loss = model.evaluate(x,y)
def evaluate(model, criterion, x, y): #가중치 갱신이 필요없어서 optimizer x
    model.eval() # 평가모드, 기울기계산, 가중치 갱신을 하지 말아라하는 의미, 드랍아웃, batch normalization도 적용 x
    
    with torch.no_grad():
        y_predict = model(x)
        loss2 = criterion(y, y_predict) #loss 최종값
    return loss2.item()

loss2=evaluate(model, criterion, x, y)
print('최종 loss : ', loss2)

results = model(torch.Tensor([[10, 1.3]]).to(DEVICE))
print('10의 예측값 :', results.item())

# 최종 loss :  5.1159079007606287e-14
# 10의 예측값 : tensor([[10.0000]])