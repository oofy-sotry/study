import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# =============================================
# 오차제곱합 (SSE, Sum of Squared Error)
# E = 1/2 * Σ(yk - tk)²
# =============================================
def sum_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)

# 정답: 숫자 2 (원-핫 인코딩)
t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])

# 예측1: 숫자 2일 확률이 가장 높음 (정답에 가까운 예측)
y1 = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])

# 예측2: 숫자 7일 확률이 가장 높음 (틀린 예측)
y2 = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])

print("=== 오차제곱합 ===")
print("정답에 가까운 예측:", sum_squared_error(y1, t))
print("틀린 예측:        ", sum_squared_error(y2, t))
print("=> 정답에 가까울수록 SSE 값이 작음")
print("==============================")

# =============================================
# 교차 엔트로피 오차 (CEE, Cross Entropy Error)
# E = -Σ tk * log(yk)
# 원-핫 인코딩에서 tk=1인 정답 인덱스의 log(yk)만 실질적으로 계산됨
# =============================================
def cross_entropy_error(y, t):
    delta = 1e-7   # log(0) = -inf 방지용 아주 작은 값
    return -np.sum(t * np.log(y + delta))

print("=== 교차 엔트로피 오차 ===")
print("정답에 가까운 예측:", cross_entropy_error(y1, t))
print("틀린 예측:        ", cross_entropy_error(y2, t))
print("=> 정답에 가까울수록 CEE 값이 작음")
print("==============================")

# =============================================
# 미니배치 학습
# =============================================
from dataset.mnist import load_mnist

x_train, t_train, x_test, t_test = load_mnist(normalize=True, flatten=True, one_hot_label=True)

print("=== 미니배치 추출 ===")
train_size = x_train.shape[0]   # 60000
batch_size = 10

batch_mask = np.random.choice(train_size, batch_size)  # 무작위 인덱스 10개 추출
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

print("전체 훈련 데이터:", train_size)
print("추출된 인덱스:  ", batch_mask)
print("x_batch shape: ", x_batch.shape)   # (10, 784)
print("t_batch shape: ", t_batch.shape)   # (10, 10)
print("==============================")

# 배치용 CEE: 합이 아닌 평균으로 계산
def cross_entropy_error_batch(y, t):
    if y.ndim == 1:                    # 이미지 1장이면 배치 형태로 변환
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(t * np.log(y + delta)) / batch_size   # 평균 손실

print("=== 배치용 CEE ===")
# 임의의 예측값으로 손실 계산 (실제로는 신경망 출력값을 사용)
y_batch = np.random.rand(batch_size, 10)
y_batch = y_batch / y_batch.sum(axis=1, keepdims=True)   # 합이 1이 되도록 정규화
print("배치 평균 손실:", cross_entropy_error_batch(y_batch, t_batch))
print("==============================")
