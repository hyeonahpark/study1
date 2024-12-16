import numpy as np
from sklearn.covariance import EllipticEnvelope

# 데이터 정의
aaa = np.array([[-10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50],
                [100, 200, -30, 400, 500, 600, -70000, 800, 900, 1000, 210, 420, 350]]).T

# 결과를 저장할 리스트
all_results = []

# 각 열에 대해 EllipticEnvelope 적용
for col_idx in range(aaa.shape[1]):
    data_column = aaa[:, col_idx].reshape(-1, 1)  # 열 단위로 데이터를 추출하고 2D로 변환
    outliers = EllipticEnvelope()  # 모델 초기화
    outliers.fit(data_column)  # 모델 학습
    results = outliers.predict(data_column)  # 이상치 예측
    all_results.append(results)  # 결과 저장

# 출력
for col_idx, result in enumerate(all_results):
    print(f"Column {col_idx + 1}: {result}")
