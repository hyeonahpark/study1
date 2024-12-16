import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용DEVICE :', DEVICE)


#1. data
x = np.array([1,2,3])
y = np.array([1,2,3])

x = torch.FloatTensor(x)
print(x.shape) #torch_size([3])
print(x.size()) #torch_size([3])

x = torch.FloatTensor(x).unsqueeze(1).to(DEVICE) # (3, ) -> (3,1)
# print(x)
y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE) # (3, ) -> (3,1)
print(x.shape, y.shape)
print(x.size(), y.size())

#2. modeling
# model = Sequential()
# model.add(Dense(1, input_dim = 1))

#model = nn.Linear(1,1).to(DEVICE) #input, output   # y= xw+b

model = nn.Sequential(
    nn.Linear(1,5),
    nn.Linear(5,4),
    nn.Linear(4,3),
    nn.Linear(3,2),
    nn.Linear(2,1),    
).to(DEVICE)


#3. compile
# model.compile(loss='mse', optizer='adam')
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x, y):
    # model.train()   #훈련모드
    optimizer.zero_grad() #각 배치마다 기울기를 초과하여, 기울기 누적에 의한 문제 해결
    #로스를 기울기로 미분한 것이 경사임! 위에서 기울기는 경사를 뜻함
    
    hypothesis = model(x) #가설. y=wx+b
    loss = criterion(hypothesis, y) #loss=mse()
    
    loss.backward() #기울기(gradient)값 계산까지. #역전파 시작
    optimizer.step() #가중치(w) 갱신              #역전파 끝
    
    return loss.item()

epochs = 2000

for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x, y)
    print('epoch: {}, loss: {}'.format(epoch, loss))  #verbose

print("===================================================================")

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

results = model(torch.Tensor([[4]]).to(DEVICE))
print('4의 예측값 :', results)