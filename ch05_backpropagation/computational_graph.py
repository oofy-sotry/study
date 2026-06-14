import numpy as np

# =============================================
# 5.1 계산 그래프와 역전파
# 핵심 공식: 역전파 = 상류에서 온 기울기 × 자신의 로컬 미분
# ∂L/∂x = ∂L/∂z × ∂z/∂x
# =============================================

# =============================================
# 예시: 사과 100원 × 2개, 세금 1.1배
# 순전파: 100 → [×2] → 200 → [×1.1] → 220
# 역전파:   ?  ← [×2] ←  ?  ← [×1.1] ←  1
# =============================================
apple_price = 100
quantity    = 2
tax         = 1.1

# 순전파
subtotal     = apple_price * quantity   # 100 × 2 = 200
total        = subtotal * tax           # 200 × 1.1 = 220

print("=== 순전파 ===")
print(f"사과 가격: {apple_price}원")
print(f"소계 (×{quantity}):  {subtotal}원")
print(f"합계 (×{tax}): {total}원")
print()

# =============================================
# 역전파: 오른쪽 → 왼쪽, 연쇄법칙 적용
# 각 단계: 상류에서 온 기울기 × 자신의 로컬 미분
# =============================================

# 출발값: ∂L/∂L = 1 (자기 자신에 대한 미분은 항상 1)
dL_dTotal = 1

# ×1.1 노드 역전파
# z = x × 1.1  =>  ∂z/∂x = 1.1
dL_dSubtotal = dL_dTotal * tax          # 1 × 1.1 = 1.1

# ×2 노드 역전파
# z = x × 2  =>  ∂z/∂x = 2
dL_dApplePrice = dL_dSubtotal * quantity  # 1.1 × 2 = 2.2

print("=== 역전파 ===")
print(f"출발값 (∂L/∂최종값):      {dL_dTotal}")
print(f"×{tax} 통과 후 (∂L/∂소계): {dL_dSubtotal}")
print(f"×{quantity} 통과 후 (∂L/∂사과가격): {dL_dApplePrice}")
print()
print(f"=> 사과 가격이 1원 오르면 최종 금액은 {dL_dApplePrice}원 오름")
print(f"   (검산: 1원 × {quantity}개 × 세금{tax} = {1 * quantity * tax}원)")
print("==============================\n")

# =============================================
# 수치 미분으로 검증 (역전파 값이 맞는지 확인)
# =============================================
def total_price(apple_price):
    return apple_price * quantity * tax

h = 1e-4
numerical = (total_price(apple_price + h) - total_price(apple_price - h)) / (2 * h)

print("=== 수치 미분으로 검증 ===")
print(f"역전파 결과:    {dL_dApplePrice}")
print(f"수치 미분 결과: {numerical}")
print(f"오차: {abs(dL_dApplePrice - numerical):.2e}")
print("=> 역전파와 수치 미분 결과가 일치함")
print("==============================\n")

# =============================================
# 사과 + 귤 예시 (변수 2개)
# 사과 100원 × 2개, 귤 150원 × 3개, 세금 1.1배
# =============================================
apple_price  = 100
apple_num    = 2
orange_price = 150
orange_num   = 3
tax          = 1.1

# 순전파
apple_total  = apple_price * apple_num      # 200
orange_total = orange_price * orange_num    # 450
subtotal     = apple_total + orange_total   # 650
total        = subtotal * tax               # 715

print("=== 사과 + 귤 순전파 ===")
print(f"사과 소계:  {apple_total}원")
print(f"귤 소계:    {orange_total}원")
print(f"합계:       {subtotal}원")
print(f"세금 포함:  {total}원")
print()

# 역전파
dL_dTotal    = 1
dL_dSubtotal = dL_dTotal * tax             # ×1.1 노드: 1 × 1.1 = 1.1

# + 노드: 덧셈은 기울기를 그대로 전달 (∂(x+y)/∂x = 1, ∂(x+y)/∂y = 1)
dL_dAppleTotal  = dL_dSubtotal * 1         # 1.1
dL_dOrangeTotal = dL_dSubtotal * 1         # 1.1

# × 노드
dL_dApplePrice  = dL_dAppleTotal * apple_num    # 1.1 × 2 = 2.2
dL_dOrangePrice = dL_dOrangeTotal * orange_num  # 1.1 × 3 = 3.3

print("=== 사과 + 귤 역전파 ===")
print(f"∂L/∂사과가격: {dL_dApplePrice}")
print(f"∂L/∂귤가격:   {dL_dOrangePrice}")
print(f"=> 사과가 1원 오르면 최종 {dL_dApplePrice}원 증가")
print(f"=> 귤이 1원 오르면 최종 {dL_dOrangePrice}원 증가")
print("==============================")
