from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor

#1. data
datasets= load_breast_cancer()
x = datasets.data
y=datasets.target

from sklearn.model_selection import train_test_split


random_state=1223

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_state, stratify=y)

#2. model
model1 = DecisionTreeRegressor(random_state=random_state)
model2 = RandomForestRegressor(random_state=random_state)
model3 = GradientBoostingRegressor(random_state=random_state)
model4 = XGBRegressor(random_state=random_state)

models = [model1, model2, model3, model4]

for model in models :
    model.fit(x_train, y_train)
    print("===================", model.__class__.__name__, "====================")
    print('acc :', model.score(x_test, y_test))
    print(model.feature_importances_)
    

import numpy as np
import matplotlib.pyplot as plt

# def plot_feature_importances_dataset(model):
#     n_features = datasets.data.shape[1]
#     plt.barh(np.arange(n_features), model.feature_importances_, align='center')
#     plt.yticks(np.arange(n_features), datasets.feature_names)
#     plt.xlabel("Feature Importances")
#     plt.ylabel("Features")
#     plt.ylim(-1, n_features)
#     plt.title(model.__class__.__name__)
    
# plot_feature_importances_dataset(model)
# plt.show()
    
from xgboost.plotting import plot_importance
plot_importance(model)
plt.show()    

#=================== DecisionTreeClassifier(random_state=777) ====================
# acc : 1.0
# [0.         0.01669139 0.39356123 0.58974738]
# =================== RandomForestClassifier(random_state=777) ====================
# acc : 1.0
# [0.09138045 0.02240597 0.43366737 0.45254621]
# =================== GradientBoostingClassifier(random_state=777) ====================
# acc : 1.0
# [0.00140476 0.01397214 0.65378516 0.33083793]
# =================== XGBClassifier(base_score=None, booster=None, callbacks=None,
#               colsample_bylevel=None, colsample_bynode=None,
#               colsample_bytree=None, device=None, early_stopping_rounds=None,
#               enable_categorical=False, eval_metric=None, feature_types=None,
#               gamma=None, grow_policy=None, importance_type=None,
#               interaction_constraints=None, learning_rate=None, max_bin=None,
#               max_cat_threshold=None, max_cat_to_onehot=None,
#               max_delta_step=None, max_depth=None, max_leaves=None,
#               min_child_weight=None, missing=nan, monotone_constraints=None,
#               multi_strategy=None, n_estimators=None, n_jobs=None,
#               num_parallel_tree=None, objective='multi:softprob', ...) ====================
# acc : 1.0
# [0.01057412 0.01290562 0.91485673 0.0616635 ]