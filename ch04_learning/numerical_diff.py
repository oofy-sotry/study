import numpy as np

# =============================================
# 수치 미분 (Numerical Differentiation)
# df(x)/dx ≈ (f(x+h) - f(x-h)) / 2h  (중앙 차분)
# =============================================
def numerical_diff(f, x):
    h = 1e-4   # h를 너무 작게 하면 라운딩 오류 발생 => 1e-4가 적절
    return (f(x + h) - f(x - h)) / (2 * h)   # 중앙 차분: 전진 차분보다 오차가 적음

# =============================================
# 테스트 함수: y = 0.01x² + 0.1x
# 해석적 미분: dy/dx = 0.02x + 0.1
# =============================================
def function_1(x):
    return 0.01 * x**2 + 0.1 * x

if __name__ == '__main__':
    print("=== 수치 미분 결과 ===")
    result_x5  = numerical_diff(function_1, 5)
    result_x10 = numerical_diff(function_1, 10)
    print("x=5  수치 미분값:", result_x5)    # 기댓값: 0.02*5  + 0.1 = 0.2
    print("x=10 수치 미분값:", result_x10)   # 기댓값: 0.02*10 + 0.1 = 0.3
    print("=> 해석적 미분값(0.2, 0.3)과 거의 일치함")
    print("==============================")

    # =============================================
    # 전진 차분 vs 중앙 차분 오차 비교
    # 전진 차분: (f(x+h) - f(x)) / h  -> x와 x+h 사이 기울기 => 오차 큼
    # 중앙 차분: (f(x+h) - f(x-h)) / 2h -> x 중심 대칭 => 오차 작음
    # =============================================
    def numerical_diff_forward(f, x):
        h = 1e-4
        return (f(x + h) - f(x)) / h   # 전진 차분

    analytic_x5 = 0.02 * 5 + 0.1   # 해석적 미분 정답

    print("=== 전진 차분 vs 중앙 차분 오차 비교 (x=5) ===")
    print("해석적 미분값:  ", analytic_x5)
    print("중앙 차분 오차:", abs(numerical_diff(function_1, 5) - analytic_x5))
    print("전진 차분 오차:", abs(numerical_diff_forward(function_1, 5) - analytic_x5))
    print("=> 중앙 차분의 오차가 더 작음")
    print("==============================")
