import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from gradient import numerical_gradient as num_grad

# =============================================
# 공통 함수
# =============================================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)   # 오버플로우 방지
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

# =============================================
# 2층 신경망 클래스 (TwoLayerNet)
# 구조: 입력층(784) → 은닉층(hidden_size, sigmoid) → 출력층(10, softmax)
# params 딕셔너리로 W1, b1, W2, b2 를 한꺼번에 관리
# =============================================
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        z1 = sigmoid(np.dot(x, W1) + b1)
        y  = softmax(np.dot(z1, W2) + b2)
        return y

    def loss(self, x, t):
        return cross_entropy_error(self.predict(x), t)

    def accuracy(self, x, t):
        y = np.argmax(self.predict(x), axis=1)
        t = np.argmax(t, axis=1)
        return np.sum(y == t) / float(x.shape[0])

    # 수치 미분으로 기울기 계산 — 정확하지만 매우 느림 (개념 확인용)
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        for key in ('W1', 'b1', 'W2', 'b2'):
            grads[key] = num_grad(loss_W, self.params[key])
        return grads

    # 역전파로 기울기 계산 — 수치 미분과 결과 동일, 속도는 훨씬 빠름 (5장 내용)
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        batch_num = x.shape[0]

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y  = softmax(a2)

        dy = (y - t) / batch_num
        grads = {}
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        dz1 = np.dot(dy, W2.T)
        da1 = sigmoid(a1) * (1 - sigmoid(a1)) * dz1
        grads['W1'] = np.dot(x.T, da1)
        grads['b1'] = np.sum(da1, axis=0)
        return grads


if __name__ == '__main__':
    from dataset.mnist import load_mnist

    # =============================================
    # 데이터 로드
    # =============================================
    x_train, t_train, x_test, t_test = load_mnist(normalize=True, flatten=True, one_hot_label=True)

    # =============================================
    # 수치 미분 기울기 확인 (소규모 — 속도 때문에 배치 1장만)
    # 실제 학습에는 역전파를 사용
    # =============================================
    print("=== 수치 미분 기울기 shape 확인 (배치 1장) ===")
    net_check = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    x_sample = x_train[:1]
    t_sample = t_train[:1]
    grads_num = net_check.numerical_gradient(x_sample, t_sample)
    for key in ('W1', 'b1', 'W2', 'b2'):
        print(f"grads['{key}'] shape:", grads_num[key].shape)
    print("==============================\n")

    # =============================================
    # 미니배치 학습 (역전파 사용)
    # 하이퍼파라미터
    # =============================================
    iters_num    = 10000
    batch_size   = 100
    learning_rate = 0.1
    train_size   = x_train.shape[0]

    # 1에폭당 반복 횟수: 전체 데이터를 배치 크기로 나눈 수
    iter_per_epoch = max(train_size // batch_size, 1)   # 600

    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    train_loss_list  = []
    train_acc_list   = []
    test_acc_list    = []

    print("=== 미니배치 학습 시작 (역전파, 10,000회) ===")
    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        grad = network.gradient(x_batch, t_batch)   # 역전파로 기울기 계산

        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]

        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        # 1에폭마다 훈련/시험 정확도 기록
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc  = network.accuracy(x_test,  t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            epoch = i // iter_per_epoch
            print(f"에폭 {epoch:2d} | 훈련 정확도: {train_acc:.4f} | 시험 정확도: {test_acc:.4f}")

    print("==============================")
    print(f"최종 훈련 정확도: {train_acc_list[-1]:.4f}")
    print(f"최종 시험 정확도: {test_acc_list[-1]:.4f}")
    print("=> 두 정확도가 비슷하면 과대적합 없이 잘 학습된 것")
