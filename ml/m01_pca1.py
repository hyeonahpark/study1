from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.decomposition import PCA

#1. data
datasets = load_iris()
x = datasets['data']
y = datasets.target

print(x.shape, y.shape) # (150, 4) (150, )

scaler = StandardScaler()
x = scaler.fit_transform(x)

pca = PCA(n_components = 2)
x = pca.fit_transform(x)


print(x)
print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(
    
    x, y, train_size=0.9, random_state = 1186, shuffle=True, stratify=y,
) #stratify=y : y의 라벨에 맞춰서 train_test_split의 비율을 정함.

#2. modeling
model = RandomForestClassifier(random_state=1186)

#3. compile
model.fit(x_train, y_train)

#4. predict
results = model.score(x_test, y_test)
print(x.shape)
print('model.score: ', results) # model.score:  1.0 -> accuracy
#RandomForestRegressor는 score 뽑으면 R2 score 뽑아줌

# (150, 4)
# model.score:  1.0

# (150, 3)
# model.score:  1.0

# (150, 2)
# model.score:  0.9333333333333333

# (150, 1)
# model.score:  1.0