from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

#1. data
x, y = load_iris(return_X_y=True)
print(x.shape, y.shape) #(150, 4) (150,)

from sklearn.model_selection import train_test_split


random_state=3

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_state)

#2. model
model1 = DecisionTreeClassifier(random_state=random_state)
model2 = RandomForestClassifier(random_state=random_state)
model3 = GradientBoostingClassifier(random_state=random_state)
model4 = XGBClassifier(random_state=random_state)

models = [model1, model2, model3, model4]

for model in models :
    model.fit(x_train, y_train)
    print("===================", model.__class__.__name__, "====================")
    print('acc :', model.score(x_test, y_test))
    print(model.feature_importances_)
    

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