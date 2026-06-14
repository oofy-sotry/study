import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


# =============================================
# 헬퍼 함수
# =============================================
def softmax(x):
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)   # 오버플로우 방지
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


# =============================================
# 5.6.1 / 5.6.2 Affine 계층 (배치 대응)
# 순전파: Y = X·W + b
# 역전파:
#   dX = dout · W^T          — shape: (N,M)·(M,D) = (N,D)
#   dW = X^T · dout          — shape: (D,N)·(N,M) = (D,M)
#   db = Σ dout (axis=0)     — 브로드캐스트의 역연산: N개분 합산 → (M,)
# x: 순전파 입력 저장 — 역전파의 dW = x.T · dout 계산에 필요
# =============================================
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
        self.db = np.sum(dout, axis=0)    # 배치 방향(axis=0)으로 합산
        return dx


# =============================================
# 5.6.3 Softmax-with-Loss 계층
# 순전파: x → softmax → y → cross_entropy_error → L
# 역전파: dx = (y - t) / batch_size
#   CEE의 역전파(-t/y)와 softmax의 역전파를 연쇄법칙으로 합치면 서로 상쇄됨
#   → y - t 라는 직관적인 형태로 단순화
# y: softmax 출력 저장 — 역전파의 y-t 계산에 필요
# =============================================
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y    = None   # softmax 출력
        self.t    = None   # 정답 레이블

    def forward(self, x, t):
        self.t    = t
        self.y    = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size   # (예측 - 정답) / 배치 크기
        return dx


# =============================================
# Affine 계층 순전파 & 역전파 확인
# =============================================
print("=== Affine 계층 ===")

x = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0]])   # shape: (2, 3) — 배치=2, 입력=3
W = np.array([[0.1, 0.2],
              [0.3, 0.4],
              [0.5, 0.6]])        # shape: (3, 2)
b = np.array([0.1, 0.2])

affine = Affine(W, b)
out = affine.forward(x)
print(f"입력 x     shape: {x.shape}")
print(f"가중치 W   shape: {W.shape}")
print(f"순전파 출력 shape: {out.shape}")
print(f"순전파 출력:\n{out}")
print()

dout = np.ones((2, 2))
dx = affine.backward(dout)
print(f"역전파 dx  shape: {dx.shape}  (입력 x와 동일)")
print(f"역전파 dW  shape: {affine.dW.shape}  (W와 동일)")
print(f"역전파 db  shape: {affine.db.shape}  (b와 동일)")
print(f"db = {affine.db}  ← 배치 2개분을 axis=0으로 합산한 결과")
print("==============================\n")


# =============================================
# Softmax-with-Loss 계층 순전파 & 역전파 확인
# =============================================
print("=== SoftmaxWithLoss 계층 ===")

x = np.array([[1.0, 2.0, 3.0],
              [3.0, 1.0, 0.5]])   # shape: (2, 3) — 배치=2, 클래스=3
t = np.array([[0, 0, 1],
              [1, 0, 0]])         # 정답 레이블 (원-핫 인코딩)

swl = SoftmaxWithLoss()
loss = swl.forward(x, t)
print(f"softmax 출력 y:\n{np.round(swl.y, 4)}")
print(f"손실 L = {loss:.4f}")
print()

dx = swl.backward()
print(f"역전파 dx = (y - t) / batch_size:\n{np.round(dx, 4)}")
print("=> 정답 클래스의 기울기: 음수 (확률을 높이는 방향)")
print("   오답 클래스의 기울기: 양수 (확률을 낮추는 방향)")
print("==============================\n")


# =============================================
# 수치 미분으로 Affine 역전파 검증
# =============================================
def numerical_gradient_2d(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            tmp = x[i, j]
            x[i, j] = tmp + h;  fxh1 = f(x)
            x[i, j] = tmp - h;  fxh2 = f(x)
            grad[i, j] = (fxh1 - fxh2) / (2 * h)
            x[i, j] = tmp
    return grad

print("=== 수치 미분으로 Affine dW 검증 ===")
np.random.seed(0)
x_v = np.random.randn(3, 4)
W_v = np.random.randn(4, 5)
b_v = np.random.randn(5)

affine2 = Affine(W_v.copy(), b_v.copy())
affine2.forward(x_v)
affine2.backward(np.ones((3, 5)))
bp_dW = affine2.dW.copy()

def loss_fn(W):
    affine2.W = W
    return np.sum(affine2.forward(x_v))   # sum이 손실이면 dout은 all-ones

nd_dW = numerical_gradient_2d(loss_fn, W_v.copy())

diff = np.max(np.abs(bp_dW - nd_dW))
print(f"dW 역전파 vs 수치 미분 최대 오차: {diff:.2e}")
print("=> 구현 정확" if diff < 1e-6 else "=> 오차 확인 필요")
print("==============================\n")


# =============================================
# 수치 미분으로 SoftmaxWithLoss 역전파 검증
# =============================================
def numerical_gradient_1d(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    for i in range(x.size):
        tmp = x.flat[i]
        x.flat[i] = tmp + h;  fxh1 = f(x)
        x.flat[i] = tmp - h;  fxh2 = f(x)
        grad.flat[i] = (fxh1 - fxh2) / (2 * h)
        x.flat[i] = tmp
    return grad

print("=== 수치 미분으로 SoftmaxWithLoss 역전파 검증 ===")
np.random.seed(1)
x_v2 = np.random.randn(2, 3)
t_v2 = np.array([[0, 0, 1], [1, 0, 0]], dtype=float)

swl2 = SoftmaxWithLoss()
swl2.forward(x_v2, t_v2)
bp_dx = swl2.backward()

def loss_fn2(x):
    return swl2.forward(x, t_v2)

nd_dx = numerical_gradient_2d(loss_fn2, x_v2.copy())

diff2 = np.max(np.abs(bp_dx - nd_dx))
print(f"dx 역전파 vs 수치 미분 최대 오차: {diff2:.2e}")
print("=> 구현 정확" if diff2 < 1e-6 else "=> 오차 확인 필요")
print("==============================")
