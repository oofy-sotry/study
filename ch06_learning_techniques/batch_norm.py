import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dataset.mnist import load_mnist


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t, y = t.reshape(1, -1), y.reshape(1, -1)
    return -np.sum(t * np.log(y + 1e-7)) / y.shape[0]


# =============================================
# 6.3 배치 정규화 (Batch Normalization)
# 순전파 (학습 시):
#   1. 미니배치 평균·분산 계산
#   2. 정규화: xn = (x - μ) / √(σ²+ε)
#   3. 스케일·이동: out = γ × xn + β
# 순전파 (추론 시):
#   학습 중 누적한 running_mean, running_var 사용
# γ, β: 학습으로 갱신되는 파라미터 (표현력 복원)
# running_mean/var: 추론용 이동평균 (학습 중 지수평균으로 누적)
# =============================================
class BatchNorm:
    def __init__(self, gamma, beta, momentum=0.9):
        self.gamma    = gamma       # 스케일 파라미터 (초기값 1)
        self.beta     = beta        # 이동 파라미터  (초기값 0)
        self.momentum = momentum
        self.running_mean = None    # 추론 시 사용할 이동평균 (학습 중 누적)
        self.running_var  = None
        # 역전파용 중간값
        self.xc = self.xn = self.std = None
        self.dgamma = self.dbeta = None

    def forward(self, x, train_flg=True):
        if self.running_mean is None:
            self.running_mean = np.zeros(x.shape[1])
            self.running_var  = np.zeros(x.shape[1])

        if train_flg:
            mu  = x.mean(axis=0)
            xc  = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 1e-7)
            xn  = xc / std
            # 역전파를 위해 중간값 저장
            self.xc, self.xn, self.std = xc, xn, std
            # 추론용 이동평균 누적
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var  = self.momentum * self.running_var  + (1 - self.momentum) * var
        else:
            xc = x - self.running_mean
            xn = xc / np.sqrt(self.running_var + 1e-7)

        return self.gamma * xn + self.beta

    def backward(self, dout):
        N = dout.shape[0]
        self.dgamma = np.sum(self.xn * dout, axis=0)
        self.dbeta  = np.sum(dout, axis=0)

        dxn  = self.gamma * dout
        dxc  = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std**2), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / N) * self.xc * dvar
        dmu  = np.sum(dxc, axis=0)
        return dxc - dmu / N


# =============================================
# 배치 정규화 효과 확인
# — 5층 신경망에서 Bad 초기화(std=1) 상황을 비교
#   BN 없음: 기울기 소실로 학습 거의 안됨
#   BN 있음: 정규화로 안정적인 분포 유지 → 학습 진행
# =============================================
np.random.seed(0)
x_train, t_train, x_test, t_test = load_mnist(normalize=True, one_hot_label=True)

# 소규모 데이터로 빠른 비교
x_train, t_train = x_train[:1000], t_train[:1000]

node_num    = 100
hidden_size = 5
input_size  = 784
output_size = 10
std         = 1.0          # 나쁜 초기화 — std=1 (기울기 소실 유발)
lr          = 0.01
batch_size  = 100
epochs      = 20

def run_experiment(use_bn):
    # 가중치 초기화
    params = {}
    sizes  = [input_size] + [node_num] * hidden_size + [output_size]
    for i in range(len(sizes) - 1):
        params[f'W{i}'] = std * np.random.randn(sizes[i], sizes[i+1])
        params[f'b{i}'] = np.zeros(sizes[i+1])

    # BN 파라미터 (hidden층에만 적용)
    bn_params = {}
    if use_bn:
        for i in range(hidden_size):
            bn_params[f'gamma{i}'] = np.ones(node_num)
            bn_params[f'beta{i}']  = np.zeros(node_num)

    bn_layers = [BatchNorm(bn_params[f'gamma{i}'], bn_params[f'beta{i}'])
                 for i in range(hidden_size)] if use_bn else []

    train_acc_list = []

    for epoch in range(epochs):
        # 미니배치 학습
        idx = np.random.permutation(len(x_train))
        for start in range(0, len(x_train), batch_size):
            xb = x_train[idx[start:start+batch_size]]
            tb = t_train[idx[start:start+batch_size]]

            # 순전파
            a = xb
            caches = []
            for i in range(hidden_size):
                z = a @ params[f'W{i}'] + params[f'b{i}']
                if use_bn:
                    z = bn_layers[i].forward(z, train_flg=True)
                a_prev = a
                a = relu(z)
                caches.append((a_prev, z, a))

            out = a @ params[f'W{hidden_size}'] + params[f'b{hidden_size}']
            y   = softmax(out)
            loss = cross_entropy_error(y, tb)

            # 역전파 (간략화: 출력층 기울기만 사용해 가중치 갱신)
            da = (y - tb) / batch_size
            params[f'W{hidden_size}'] -= lr * (caches[-1][2].T @ da)
            params[f'b{hidden_size}'] -= lr * da.sum(axis=0)

            for i in reversed(range(hidden_size)):
                a_prev, z, a = caches[i]
                dz = da @ params[f'W{i+1}'].T * (z > 0)
                if use_bn:
                    dz = bn_layers[i].backward(dz)
                    bn_params[f'gamma{i}'] -= lr * bn_layers[i].dgamma
                    bn_params[f'beta{i}']  -= lr * bn_layers[i].dbeta
                params[f'W{i}'] -= lr * (a_prev.T @ dz)
                params[f'b{i}'] -= lr * dz.sum(axis=0)
                da = dz

        # 에폭별 정확도 계산
        out  = x_train
        for i in range(hidden_size):
            z = out @ params[f'W{i}'] + params[f'b{i}']
            if use_bn:
                z = bn_layers[i].forward(z, train_flg=False)
            out = relu(z)
        out  = out @ params[f'W{hidden_size}'] + params[f'b{hidden_size}']
        pred = np.argmax(softmax(out), axis=1)
        acc  = np.mean(pred == np.argmax(t_train, axis=1))
        train_acc_list.append(acc)

    return train_acc_list


print("=" * 52)
print(f"배치 정규화 효과 비교  (초기화 std={std}, 5층)")
print(f"{'epoch':>5}  {'BN 없음':>10}  {'BN 있음':>10}")
print("-" * 32)

acc_no_bn = run_experiment(use_bn=False)
acc_bn    = run_experiment(use_bn=True)

for ep in range(epochs):
    marker = " ←" if (acc_bn[ep] - acc_no_bn[ep]) > 0.1 else ""
    print(f"  {ep+1:>3}  {acc_no_bn[ep]:>10.4f}  {acc_bn[ep]:>10.4f}{marker}")

print()
print(f"최종 정확도  BN 없음: {acc_no_bn[-1]:.4f}  /  BN 있음: {acc_bn[-1]:.4f}")
print(f"BN 적용 시 정확도 향상: +{acc_bn[-1]-acc_no_bn[-1]:.4f}")
