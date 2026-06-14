import numpy as np

def step_function(x):
    if x > 0:
        return 1
    else:
        return 0

print("step_function")
print(step_function(-1))
print(step_function(-2))
print(step_function(1))
print(step_function(2))
print("==============================")

# 넘파이 배열을 지원하기 위한 방법
def step_function2(x):
    x = np.array(x)
    y = x > 0
    return  y.astype(int)

print("step_function2")
print(step_function2([1.0, 2.0]))
print(step_function2([-1.0, 2.0]))
print(step_function2([1.0, -2.0]))
print(step_function2([-1.0, -2.0]))
print("==============================")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

print("sigmoid")
print(sigmoid(0))
print(sigmoid(1.0))
print(sigmoid(-1.0))
print(sigmoid(np.array([-1.0, 0.0, 1.0])))
print("==============================")

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

# 오버플로우 방지 버전
def softmax2(x):
    c = np.max(x)
    exp_x = np.exp(x - c)
    return exp_x / np.sum(exp_x)

print("softmax")
a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y)
print("합계:", np.sum(y))
print("==============================")

print("softmax2 (오버플로우 방지)")
y2 = softmax2(a)
print(y2)
print("합계:", np.sum(y2))
print("==============================")