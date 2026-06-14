import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pickle
from dataset.mnist import load_mnist


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    c = np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x - c)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def load_network():
    weight_path = os.path.join(os.path.dirname(__file__), 'sample_weight.pkl')
    with open(weight_path, 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y  = softmax(a3)
    return y


# 데이터 로드
x_train, t_train, x_test, t_test = load_mnist(normalize=True, flatten=True)

print("=== 데이터 형태 ===")
print("훈련 이미지:", x_train.shape)
print("테스트 이미지:", x_test.shape)

# 가중치 로드
network = load_network()

# 배치 처리로 정확도 계산
batch_size = 100
accuracy_cnt = 0

for i in range(0, len(x_test), batch_size):
    x_batch = x_test[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)  # 가장 높은 확률의 인덱스 = 예측 숫자
    accuracy_cnt += np.sum(p == t_test[i:i+batch_size])

print("\n=== 결과 ===")
print(f"정확도: {accuracy_cnt / len(x_test) * 100:.2f}%")

# 개별 예측 예시
print("\n=== 개별 예측 예시 (처음 5장) ===")
y = predict(network, x_test[:5])
pred = np.argmax(y, axis=1)
for i in range(5):
    print(f"예측: {pred[i]}  정답: {t_test[i]}  {'O' if pred[i] == t_test[i] else 'X'}")
