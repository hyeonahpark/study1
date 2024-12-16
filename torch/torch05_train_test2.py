import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용 DEVICE :', DEVICE)


#1. data
x = np.array(range(100)) 
y = np.array(range(1, 101)) 
print(x.shape, y.shape) #(100,) (100,)
x = torch.FloatTensor(x).unsqueeze(1).to(DEVICE)
y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)

x_pre = np.array([101,102])
x_pre = torch.FloatTensor(x_pre).unsqueeze(1).to(DEVICE)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8)

################실습########################

model = nn.Sequential(
    nn.Linear(1,5),
    nn.Linear(5,4),
    nn.Linear(4,3),
    nn.Linear(3,2),
    nn.Linear(2,1),
    # nn.Linear(5,4),
    # nn.Linear(4,3),
    # nn.Linear(3,2),
    # nn.Linear(2,1)
).to(DEVICE)

#3. compile
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)

def train(model, criterion, optimizer, x, y):
    optimizer.zero_grad()
    hypothesis=model(x)
    loss=criterion(hypothesis, y)
    loss.backward()
    optimizer.step()
    
    return loss.item()


epochs = 3000

for epoch in range(1, epochs+1):
    loss=train(model, criterion, optimizer, x_train, y_train)
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

loss2=evaluate(model, criterion, x_test, y_test)
print('최종 loss : ', loss2)

results = model(x_pre)
print('101,102의 예측값 :', results.detach().cpu().numpy())

# 최종 loss :  5.7980287238024175e-12
# 101,102의 예측값 : [[102.]
#  [103.]]