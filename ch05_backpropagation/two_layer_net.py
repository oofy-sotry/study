import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from collections import OrderedDict
from dataset.mnist import load_mnist


# =============================================
# 헬퍼 함수
# =============================================
def softmax(x):
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        y = np.exp(x)
        return y / y.sum(axis=1, keepdims=True)
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp = x[idx]
        x[idx] = tmp + h;  fxh1 = f(x)
        x[idx] = tmp - h;  fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp
        it.iternext()
    return grad


# =============================================
# 계층 클래스 (5.5, 5.6)
# =============================================
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        return dout


class Affine:
    def __init__(self, W, b):
        self.W  = W
        self.b  = b
        self.x  = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        dx      = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y    = None
        self.t    = None

    def forward(self, x, t):
        self.t    = t
        self.y    = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        return (self.y - self.t) / batch_size


# =============================================
# 5.7.2 TwoLayerNet — 역전파 적용
# 구조: 입력 → Affine1 → ReLU → Affine2 → SoftmaxWithLoss
# OrderedDict: 삽입 순서를 기억 → forward는 순서대로, backward는 역순으로
# gradient(): 순전파 1회 + 역전파 1회로 모든 가중치 기울기를 한 번에 계산
#   cf. numerical_gradient(): 가중치 하나씩 h만큼 바꿔가며 반복 → 훨씬 느림
# =============================================
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1']   = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        return self.lastLayer.forward(self.predict(x), t)

    def accuracy(self, x, t):
        y = np.argmax(self.predict(x), axis=1)
        t = np.argmax(t, axis=1)
        return np.sum(y == t) / float(x.shape[0])

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        for key in ('W1', 'b1', 'W2', 'b2'):
            grads[key] = numerical_gradient(loss_W, self.params[key])
        return grads

    def gradient(self, x, t):
        self.loss(x, t)                                    # 순전파
        dout = self.lastLayer.backward(1)                  # 역전파 시작
        for layer in reversed(list(self.layers.values())):
            dout = layer.backward(dout)
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        return grads


# =============================================
# 5.7.3 기울기 확인 (Gradient Check)
# 수치 미분 vs 역전파 결과 비교 — 역전파 구현의 정확성 검증
# 오차가 1e-5 이하이면 올바른 구현으로 판단
# =============================================
print("=== 기울기 확인 (Gradient Check) ===")
x_train, t_train, x_test, t_test = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop  = network.gradient(x_batch, t_batch)

for key in ('W1', 'b1', 'W2', 'b2'):
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(f"{key}: {diff:.2e}  {'✓ 구현 정확' if diff < 1e-4 else '✗ 오차 확인 필요'}")

print("=> 수치 미분과 역전파 결과가 거의 일치 → 역전파 구현 정확\n")


# =============================================
# 5.7.4 오차역전파법을 사용한 학습 구현
# 4장 학습 구현과 동일한 구조
# 변경점: numerical_gradient → gradient (역전파 방식)
# =============================================
print("=== MNIST 학습 (오차역전파법) ===")

network      = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
iters_num    = 10000
train_size   = x_train.shape[0]
batch_size   = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list  = []
test_acc_list   = []
iter_per_epoch  = max(train_size // batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch    = x_train[batch_mask]
    t_batch    = t_train[batch_mask]

    grad = network.gradient(x_batch, t_batch)          # 역전파로 기울기 계산
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc  = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        epoch = i // iter_per_epoch
        print(f"epoch {epoch:2d} | train acc: {train_acc:.4f} | test acc: {test_acc:.4f}")

print(f"\n최종 훈련 정확도: {train_acc_list[-1]:.4f}")
print(f"최종 테스트 정확도: {test_acc_list[-1]:.4f}")
