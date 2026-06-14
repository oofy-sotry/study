import numpy as np

# =============================================
# 기울기 (Gradient)
# 모든 변수의 편미분을 벡터로 정리한 것
# ∇f = (∂f/∂x0, ∂f/∂x1, ...)
# =============================================
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)       # x와 같은 shape의 0 배열

    # np.nditer: 다차원 배열(1D, 2D, ...)을 원소 하나씩 순회
    # multi_index로 현재 위치(인덱스)를 추적, readwrite로 값 수정 허용
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]

        x[idx] = tmp_val + h
        fxh1 = f(x)               # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)               # f(x-h)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val          # 원래 값으로 복원
        it.iternext()

    return grad

# =============================================
# 테스트 함수: f(x0, x1) = x0² + x1²
# 해석적 기울기: ∇f = (2x0, 2x1)
# =============================================
def function_2(x):
    return x[0]**2 + x[1]**2

print("=== 기울기 계산 (f(x0, x1) = x0² + x1²) ===")
print("(x0=3, x1=4) :", numerical_gradient(function_2, np.array([3.0, 4.0])))  # [6.0, 8.0]
print("(x0=0, x1=2) :", numerical_gradient(function_2, np.array([0.0, 2.0])))  # [0.0, 4.0]
print("(x0=3, x1=0) :", numerical_gradient(function_2, np.array([3.0, 0.0])))  # [6.0, 0.0]
print("=> 기울기 = 각 점에서 함수값이 가장 가파르게 증가하는 방향")
print("=> 기울기의 반대 방향으로 이동하면 함수값이 가장 빠르게 감소 (경사하강법의 원리)")
print("==============================")
