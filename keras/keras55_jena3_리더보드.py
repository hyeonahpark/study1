import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Flatten, GRU
from sklearn.metrics import r2_score, mean_squared_error


학생csv = 'jena_박현아.csv'

path1 = 'C:\\ai5\\_data\\kaggle\\jena\\'
path2 = 'C:\\ai5\\_save\\keras55\\'

datasets = pd.read_csv(path1 + 'jena_climate_2009_2016.csv', index_col = 0)

y_정답 = datasets.iloc[-144:, 1] #loc : 인덱스 위주, iloc : ?
# print(y_정답)
# print(y_정답.shape)


학생꺼 = pd.read_csv(path2 + 학생csv, index_col=0)
# print(학생꺼)


# print(y_정답[:5])
# print(학생꺼[:5])

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_정답, 학생꺼)
print("RMSE :", rmse)