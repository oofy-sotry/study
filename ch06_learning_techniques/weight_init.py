import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)


# =============================================
# 활성화값 분포 통계 출력 헬퍼
# 활성화값이 어디에 몰려 있는지를 숫자로 확인
#   near_zero : 0.1 미만 비율 (기울기 소실 위험)
#   near_one  : 0.9 초과 비율 (기울기 소실 위험)
#   saturated : near_zero + near_one (포화 비율)
#   mean/std  : 0.5 근처 / std가 클수록 고른 분포가 이상적
# =============================================
def sigmoid_stats(activations, label):
    print(f"  [{label}]")
    for i, a in activations.items():
        flat      = a.flatten()
        near_zero = np.mean(flat < 0.1)
        near_one  = np.mean(flat > 0.9)
        print(f"  layer {i+1} | mean={flat.mean():.3f}  std={flat.std():.3f}"
              f"  포화율={near_zero+near_one:.1%}  (≈0: {near_zero:.1%}, ≈1: {near_one:.1%})")
    print()

def relu_stats(activations, label):
    print(f"  [{label}]")
    for i, a in activations.items():
        flat = a.flatten()
        print(f"  layer {i+1} | mean={flat.mean():.3f}  std={flat.std():.3f}"
              f"  죽은뉴런(=0): {np.mean(flat == 0):.1%}")
    print()


# =============================================
# 6.2.2 Sigmoid — 초깃값 방식별 활성화값 분포
# 입력: 1000개 × 100차원 / 은닉층 5개 × 100노드
# =============================================
np.random.seed(0)
input_data  = np.random.randn(1000, 100)
node_num    = 100
hidden_size = 5

print("=" * 58)
print("Sigmoid — 초깃값 방식별 활성화값 분포")
print("(포화율↓ std↑ 일수록 고른 분포 = 이상적)")
print("=" * 58)

for label, std in [("std=1   → 기울기 소실", 1.0),
                   ("std=0.01 → 표현력 저하", 0.01),
                   ("Xavier  std=1/√n → 권장", None)]:
    activations = {}
    x = input_data.copy()
    for i in range(hidden_size):
        if i != 0:
            x = activations[i - 1]
        scale = np.sqrt(1.0 / node_num) if std is None else std
        w = np.random.randn(node_num, node_num) * scale
        activations[i] = sigmoid(np.dot(x, w))
    sigmoid_stats(activations, label)


# =============================================
# 6.2.3 ReLU — Xavier vs He 초기화 비교
# ReLU는 음수 입력을 0으로 만들어 신호 절반이 소멸
# → Xavier(1/√n)는 부족 → He(√(2/n))가 필요
# =============================================
print("=" * 58)
print("ReLU — Xavier vs He 초기화 비교")
print("(std가 층을 거쳐도 유지될수록 이상적)")
print("=" * 58)

for label, scale_fn in [
    ("Xavier  std=1/√n  → ReLU엔 부적합", lambda n: np.sqrt(1.0 / n)),
    ("He      std=√(2/n) → 권장",          lambda n: np.sqrt(2.0 / n)),
]:
    activations = {}
    x = input_data.copy()
    for i in range(hidden_size):
        if i != 0:
            x = activations[i - 1]
        w = np.random.randn(node_num, node_num) * scale_fn(node_num)
        activations[i] = relu(np.dot(x, w))
    relu_stats(activations, label)


# =============================================
# 초기화 방법 요약
# =============================================
print("=" * 58)
print("초기화 방법 요약")
print("=" * 58)
print("  활성화 함수        권장 초기화    표준편차")
print("  Sigmoid / tanh    Xavier        1 / √n")
print("  ReLU              He            √(2 / n)")
print()
print("  초깃값 0 또는 균일값 → 절대 사용 금지")
print("  이유: 역전파에서 모든 가중치가 똑같이 갱신됨 (대칭 문제)")
