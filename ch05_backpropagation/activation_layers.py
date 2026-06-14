import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# =============================================
# 5.5.1 ReLU 레이어
# 순전파: y = x (x>0), y = 0 (x<=0)
# 역전파: x>0이면 기울기 그대로, x<=0이면 기울기 차단(0)
# mask: 순전파 때 x<=0인 위치를 기억해뒀다가 역전파 때 활용
# =============================================
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)   # x<=0인 위치를 True로 저장
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0    # 순전파 때 0이었던 위치는 역전파 기울기도 0
        return dout


# =============================================
# 5.5.2 Sigmoid 레이어
# 순전파: y = 1 / (1 + exp(-x))
# 역전파: dL/dx = dout × y × (1-y)
# out: 순전파 출력값(y) 저장 — 역전파의 y×(1-y) 계산에 필요
# 유도: 4단계(부호반전→exp→+1→역수) 연쇄법칙 적용 후 exp(-x)=1/y-1 로 치환하면 단순화됨
# =============================================
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, dout):
        return dout * self.out * (1 - self.out)


# =============================================
# ReLU 순전파 & 역전파 확인
# =============================================
print("=== ReLU 레이어 ===")
relu = Relu()

x = np.array([1.0, -2.0, 3.0, -4.0, 0.0])
out = relu.forward(x)
print(f"입력  x   : {x}")
print(f"순전파 out: {out}")
print(f"mask(x<=0): {relu.mask}")
print()

dout = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
dx = relu.backward(dout)
print(f"상류 기울기 dout: {dout}")
print(f"역전파 dx       : {dx}")
print("=> x>0 위치만 기울기 통과, x<=0 위치는 0으로 차단")
print("==============================\n")

# =============================================
# Sigmoid 순전파 & 역전파 확인
# =============================================
print("=== Sigmoid 레이어 ===")
sigmoid = Sigmoid()

x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
out = sigmoid.forward(x)
print(f"입력  x   : {x}")
print(f"순전파 out: {np.round(out, 4)}")
print()

dout = np.ones_like(x)
dx = sigmoid.backward(dout)
print(f"상류 기울기 dout: {dout}")
print(f"역전파 dx       : {np.round(dx, 4)}")
print("=> dx = dout × y × (1-y)  — y가 0.5일 때 기울기 최대(0.25)")
print("==============================\n")

# =============================================
# 수치 미분으로 검증
# =============================================
def relu_func(x):
    return np.maximum(0, x)

def sigmoid_func(x):
    return 1 / (1 + np.exp(-x))

def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)

print("=== 수치 미분으로 검증 (x=1.5) ===")
x_val = 1.5

relu2   = Relu()
relu2.forward(np.array([x_val]))
bp_relu = relu2.backward(np.array([1.0]))[0]
nd_relu = numerical_diff(relu_func, x_val)
print(f"ReLU    — 역전파: {bp_relu}, 수치 미분: {nd_relu:.4f}")

sigmoid2 = Sigmoid()
sigmoid2.forward(np.array([x_val]))
bp_sig   = sigmoid2.backward(np.array([1.0]))[0]
nd_sig   = numerical_diff(sigmoid_func, x_val)
print(f"Sigmoid — 역전파: {bp_sig:.6f}, 수치 미분: {nd_sig:.6f}")
print("=> 역전파와 수치 미분 결과 일치")
print("==============================\n")

# =============================================
# ReLU vs Sigmoid 기울기 비교
# x가 음수일 때 두 함수의 역전파 차이
# =============================================
print("=== ReLU vs Sigmoid 기울기 비교 (x=-2) ===")
x_neg = np.array([-2.0])

relu3 = Relu()
relu3.forward(x_neg)
dx_relu = relu3.backward(np.array([1.0]))[0]

sig3 = Sigmoid()
sig3.forward(x_neg)
dx_sig = sig3.backward(np.array([1.0]))[0]

print(f"x = {x_neg[0]}")
print(f"ReLU 역전파:    {dx_relu}  (음수 입력 → 기울기 완전 차단)")
print(f"Sigmoid 역전파: {dx_sig:.6f}  (음수 입력도 기울기 약하게 전달)")
print("==============================")
