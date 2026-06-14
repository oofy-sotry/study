import numpy as np

x = np.array([1.0, 2.0, 3.0])
print(x)
print(type(x))

y = np.array([2.0, 4.0, 6.0])
print(x + y)
print(x - y)
print(x / y)

# 브로드 캐스트란?
# 크기가 다른 배열끼리 연산할 때, 크기가 1인 쪽을 늘려서 맞춰주는 것

A = np.array([[1, 2],[3, 4]])
print(A)
print(A.shape)
print(A.dtype)


B = np.array([[3, 0],[0, 6]])
print(A + B)
print(A * B)
print(A * 10)