import numpy as np
from gradient import numerical_gradient

# =============================================
# 소프트맥스 & 교차 엔트로피 오차 (3장, 4.2에서 학습한 내용)
# =============================================
def softmax(x):
    x = x - np.max(x)   # 오버플로우 방지
    return np.exp(x) / np.sum(np.exp(x))

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

# =============================================
# SimpleNet : 신경망에서의 기울기 (∂L/∂W) 계산
# 입력(2) → 출력(3) 의 단순 신경망
# 목적: 손실 L을 가중치 W로 미분해 "W를 어떻게 바꿔야 손실이 줄어드는지" 파악
# =============================================
class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)   # 가중치 (2×3, 정규분포 초기화)

    def predict(self, x):
        return np.dot(x, self.W)          # 입력 x와 가중치 W의 행렬 곱

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)                    # 확률로 변환
        return cross_entropy_error(y, t)  # 손실값 계산

# =============================================
# 신경망 기울기 계산
# =============================================
np.random.seed(42)   # 재현성을 위한 시드 고정
net = SimpleNet()

x = np.array([0.6, 0.9])   # 입력값
t = np.array([0, 0, 1])    # 정답 레이블 (숫자 2, 원-핫 인코딩)

print("=== SimpleNet 가중치 W (2×3) ===")
print(net.W)
print()

print("=== 예측값 (x · W) ===")
p = net.predict(x)
print(p)
print("가장 높은 점수의 인덱스 (예측 클래스):", np.argmax(p))
print()

print("=== 손실값 ===")
print(net.loss(x, t))
print()

# f = lambda w: net.loss(x, t)
# w는 형식 인수 — numerical_gradient가 net.W를 직접 수정하므로 실제로 사용되지 않음
f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)

print("=== ∂L/∂W : 손실 함수의 가중치에 대한 기울기 (2×3) ===")
print(dW)
print()
print("=> dW의 shape:", dW.shape, "— W의 shape와 동일")
print("=> 양수 원소: 해당 가중치를 키우면 손실 증가 (줄여야 함)")
print("=> 음수 원소: 해당 가중치를 키우면 손실 감소 (키워야 함)")
print("==============================")
