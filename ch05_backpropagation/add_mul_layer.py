import numpy as np

# =============================================
# 5.3 역전파 (Backpropagation)
# 각 노드는 순전파/역전파 두 메서드를 가짐
# forward : 순전파 계산 + 입력값 저장
# backward: 상류 기울기(dout)를 받아 하류로 전달할 기울기 반환
# =============================================

# =============================================
# 덧셈 노드
# z = x + y
# ∂z/∂x = 1,  ∂z/∂y = 1
# 역전파: 상류 기울기를 그대로 흘려보냄
# =============================================
class AddLayer:
    def forward(self, x, y):
        return x + y

    def backward(self, dout):
        # dout: 상류에서 온 기울기 (∂L/∂z)
        # 덧셈 노드의 로컬 미분이 1이므로 dout × 1 = dout 그대로 전달
        dx = dout * 1
        dy = dout * 1
        return dx, dy

# =============================================
# 곱셈 노드
# z = x × y
# ∂z/∂x = y,  ∂z/∂y = x
# 역전파: 순전파 입력값을 서로 교환해서 곱함 → 입력값 저장 필요
# =============================================
class MulLayer:
    def __init__(self):
        self.x = None   # 역전파 때 쓰기 위해 순전파 입력값 저장
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        return x * y

    def backward(self, dout):
        # dout: 상류에서 온 기울기 (∂L/∂z)
        # x쪽: dout × y (상대방 y를 곱함)
        # y쪽: dout × x (상대방 x를 곱함)
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy


# =============================================
# 덧셈 노드 역전파 확인
# =============================================
print("=== 덧셈 노드 역전파 ===")
add = AddLayer()

x, y = 3.0, 7.0
z = add.forward(x, y)
print(f"순전파: {x} + {y} = {z}")

dout = 1.3   # 상류에서 온 기울기 (맨 마지막 노드가 아니라면 1이 아닐 수 있음)
dx, dy = add.backward(dout)
print(f"역전파: 상류 기울기={dout} → ∂L/∂x={dx}, ∂L/∂y={dy}")
print("=> 덧셈 노드는 상류 기울기를 그대로 전달")
print("==============================\n")

# =============================================
# 곱셈 노드 역전파 확인
# =============================================
print("=== 곱셈 노드 역전파 ===")
mul = MulLayer()

x, y = 10.0, 5.0
z = mul.forward(x, y)
print(f"순전파: {x} × {y} = {z}")

dout = 1.3
dx, dy = mul.backward(dout)
print(f"역전파: 상류 기울기={dout} → ∂L/∂x={dx} (={dout}×{y}), ∂L/∂y={dy} (={dout}×{x})")
print("=> 곱셈 노드는 순전파 입력값을 서로 교환해서 곱함")
print("==============================\n")

# =============================================
# 사과 + 귤 쇼핑 예시 (순전파 & 역전파 전체 흐름)
# 사과: 100원 × 2개 / 귤: 150원 × 3개 / 세금: 1.1배
# =============================================
print("=== 사과 + 귤 쇼핑 순전파 ===")

apple_price  = 100
apple_num    = 2
orange_price = 150
orange_num   = 3
tax          = 1.1

mul_apple  = MulLayer()
mul_orange = MulLayer()
add_fruit  = AddLayer()
mul_tax    = MulLayer()

# 순전파
apple_total  = mul_apple.forward(apple_price, apple_num)    # 200
orange_total = mul_orange.forward(orange_price, orange_num) # 450
subtotal     = add_fruit.forward(apple_total, orange_total) # 650
total        = mul_tax.forward(subtotal, tax)               # 715

print(f"사과 소계: {apple_total}원")
print(f"귤 소계:   {orange_total}원")
print(f"합계:      {subtotal}원")
print(f"세금 포함: {total}원")
print()

print("=== 역전파 ===")
# 역전파 — 오른쪽에서 왼쪽 순서로
dout = 1.0   # 맨 마지막 노드의 출발값은 1

d_subtotal, d_tax        = mul_tax.backward(dout)
d_apple_total, d_orange_total = add_fruit.backward(d_subtotal)
d_apple_price, d_apple_num   = mul_apple.backward(d_apple_total)
d_orange_price, d_orange_num = mul_orange.backward(d_orange_total)

print(f"∂L/∂사과가격  = {d_apple_price}")   # 2.2
print(f"∂L/∂사과개수  = {d_apple_num}")     # 110
print(f"∂L/∂귤가격    = {d_orange_price}")  # 3.3
print(f"∂L/∂귤개수    = {d_orange_num}")    # 165
print(f"∂L/∂세율      = {d_tax}")           # 650
print()
print("=> 사과 가격이 1원 오르면 총액은 2.2원 오름 (2개 × 1.1세율)")
print("=> 귤 가격이 1원 오르면 총액은 3.3원 오름 (3개 × 1.1세율)")
print("==============================")
