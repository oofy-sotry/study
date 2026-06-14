import numpy as np
from gradient import numerical_gradient

# =============================================
# 경사하강법 (Gradient Descent)
# 기울기의 반대 방향으로 조금씩 이동해 함수의 최솟값을 찾는 방법
# 수식: x ← x - η * ∂f/∂x  (η: 학습률)
# =============================================
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x.copy()
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad   # 기울기 반대 방향으로 lr만큼 이동
    return x

# =============================================
# 테스트 함수: f(x0, x1) = x0² + x1²
# 최솟값: (0, 0)
# =============================================
def function_2(x):
    return x[0]**2 + x[1]**2

init_x = np.array([-3.0, 4.0])

# =============================================
# 학습률에 따른 결과 비교
# =============================================
print("=== 학습률에 따른 경사하강법 결과 ===")

result = gradient_descent(function_2, init_x, lr=0.1, step_num=100)
print("lr=0.1  (적절) :", result)            # 거의 [0, 0]에 수렴

result_large = gradient_descent(function_2, init_x, lr=10.0, step_num=100)
print("lr=10.0 (너무 큼) :", result_large)   # 발산

result_small = gradient_descent(function_2, init_x, lr=1e-10, step_num=100)
print("lr=1e-10 (너무 작음):", result_small) # 거의 이동 못함

print()
print("=> lr이 너무 크면 최솟값을 지나쳐 발산, 너무 작으면 수렴이 느림")
print("=> 학습률은 사람이 직접 설정하는 하이퍼파라미터")
print("==============================")
