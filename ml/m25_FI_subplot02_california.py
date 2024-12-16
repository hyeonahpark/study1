from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor

#1. data
datasets= fetch_california_housing()
x = datasets.data
y=datasets.target

from sklearn.model_selection import train_test_split


random_state=1223

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_state)

#2. model
model1 = DecisionTreeRegressor(random_state=random_state)
model2 = RandomForestRegressor(random_state=random_state)
model3 = GradientBoostingRegressor(random_state=random_state)
model4 = XGBRegressor(random_state=random_state)

models = [model1, model2, model3, model4]

i=0
for model in models :
    model.fit(x_train, y_train)
    print("===================", model.__class__.__name__, "====================")
    print('acc :', model.score(x_test, y_test))
    print(model.feature_importances_)
    import numpy as np
    import matplotlib.pyplot as plt
    def plot_feature_importances_dataset(model):
        n_features = datasets.data.shape[1]
        plt.barh(np.arange(n_features), model.feature_importances_, align='center')
        plt.yticks(np.arange(n_features), datasets.feature_names)
        plt.xlabel("Feature Importances")
        plt.ylabel("Features")
        plt.ylim(-1, n_features)
        plt.title(model.__class__.__name__)
    
   
    plt.subplot(2, 2, i+1)
    plot_feature_importances_dataset(model)
    i=i+1
plt.tight_layout()  # 간격 안겹치게 
plt.show()



# def plot_feature_importances_dataset(model):
#     n_features = datasets.data.shape[1]
#     plt.barh(np.arange(n_features), model.feature_importances_, align='center')
#     plt.yticks(np.arange(n_features), datasets.feature_names)
#     plt.xlabel("Feature Importances")
#     plt.ylabel("Features")
#     plt.ylim(-1, n_features)
#     plt.title(model.__class__.__name__)
    
# plot_feature_importances_dataset(model)

# for i in range(4):
#     plt.subplot(2, 2, i+1)
#     plt.imshow(models[i])
    
# plt.show()
