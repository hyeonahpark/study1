import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용 DEVICE :', DEVICE)


#1. data
x_train = np.array([1,2,3,4,5,6,7]) #(7,)
y_train = np.array([1,2,3,4,5,6,7]) # (7,)
x_test = np.array([8,9,10,11]) #(4,)
y_test = np.array([8,9,10,11]) #(4,)

print(x_train.shape, x_test.shape)


x_pre = np.array([12,13,14])

x_train = torch.FloatTensor(x_train).unsqueeze(1).to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
x_test = torch.FloatTensor(x_test).unsqueeze(1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)
x_pre = torch.FloatTensor(x_pre).unsqueeze(1).to(DEVICE)

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
print('12,13,14의 예측값 :', results.detach().cpu().numpy())

# 최종 loss :  4.547473508864641e-13
# 12,13,14의 예측값 : [[12.      ]
#  [12.999999]
#  [13.999999]]