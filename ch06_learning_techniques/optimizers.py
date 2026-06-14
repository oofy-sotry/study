import numpy as np
from collections import OrderedDict


# =============================================
# 6.1.2 SGD
# W ← W - lr × ∂L/∂W
# 단순하지만 비등방성 함수에서 지그재그 탐색 → 비효율적
# =============================================
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


# =============================================
# 6.1.4 Momentum
# v ← αv - lr × ∂L/∂W
# W ← W + v
# v(속도): 이전 이동 방향을 기억 — 관성 효과로 SGD보다 진동이 줄고 빠르게 수렴
# α(모멘텀 계수): 속도를 얼마나 유지할지 결정 (보통 0.9)
# =============================================
class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {key: np.zeros_like(val) for key, val in params.items()}
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


# =============================================
# 6.1.5 AdaGrad
# h ← h + (∂L/∂W)²
# W ← W - lr × (1/√h) × ∂L/∂W
# h(기울기 제곱 누적합): 많이 갱신된 가중치는 학습률을 낮춤 → 개별 적응형 학습률
# 단점: h가 무한히 커져 결국 학습률이 0에 수렴 → 학습 정지
# =============================================
class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {key: np.zeros_like(val) for key, val in params.items()}
        for key in params.keys():
            self.h[key] += grads[key] ** 2
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


# =============================================
# 6.1.6 Adam
# Momentum(관성) + AdaGrad(개별 학습률)의 장점을 결합
# m: 1차 모멘트 (기울기 지수이동평균)       ← Momentum 역할
# v: 2차 모멘트 (기울기 제곱 지수이동평균)  ← AdaGrad 역할
# 편향 보정: m, v가 0 초기화로 인한 초기 편향을 lr_t 로 보정
#   lr_t = lr × √(1-β₂ᵗ) / (1-β₁ᵗ)  → t가 커질수록 1에 수렴
# =============================================
class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m = {key: np.zeros_like(val) for key, val in params.items()}
            self.v = {key: np.zeros_like(val) for key, val in params.items()}
        self.iter += 1
        lr_t = self.lr * np.sqrt(1 - self.beta2**self.iter) / (1 - self.beta1**self.iter)
        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)


# =============================================
# 옵티마이저 비교 — f(x, y) = x²/20 + y²
# 최솟값: (0, 0)  /  시작점: (-7, 2)
# x방향 기울기(x/10)가 y방향(2y)보다 훨씬 작은 비등방성 함수
# → SGD는 지그재그로 탐색, 나머지는 개선된 경로를 보임
# =============================================
def f(x, y):
    return x**2 / 20.0 + y**2

def df(x, y):
    return x / 10.0, 2.0 * y

optimizers = OrderedDict([
    ("SGD",      SGD(lr=0.95)),
    ("Momentum", Momentum(lr=0.1)),
    ("AdaGrad",  AdaGrad(lr=1.5)),
    ("Adam",     Adam(lr=0.3)),
])

print("=== 옵티마이저 비교: f(x,y) = x²/20 + y²  (시작점: x=-7, y=2) ===\n")

for name, optimizer in optimizers.items():
    x, y = -7.0, 2.0
    params = {'x': x, 'y': y}

    print(f"[{name}]")
    print(f"  {'iter':>4}  {'x':>8}  {'y':>8}  {'f(x,y)':>10}")
    print(f"  {'----':>4}  {'--------':>8}  {'--------':>8}  {'----------':>10}")

    for i in range(30):
        if i % 5 == 0:
            print(f"  {i:>4}  {params['x']:>8.4f}  {params['y']:>8.4f}  {f(params['x'], params['y']):>10.6f}")
        grads = {}
        grads['x'], grads['y'] = df(params['x'], params['y'])
        optimizer.update(params, grads)

    print(f"  {30:>4}  {params['x']:>8.4f}  {params['y']:>8.4f}  {f(params['x'], params['y']):>10.6f}")
    dist = np.sqrt(params['x']**2 + params['y']**2)
    print(f"  최종 원점까지 거리: {dist:.6f}\n")
