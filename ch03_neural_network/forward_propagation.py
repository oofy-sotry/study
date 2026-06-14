import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def identity_function(x):
    return x


# 가중치와 편향 초기화
def init_network():
    network = {}

    # 1층: 입력(2) -> 은닉(3)
    network['W1'] = np.array([[0.1, 0.3, 0.5],
                               [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])

    # 2층: 은닉(3) -> 은닉(2)
    network['W2'] = np.array([[0.1, 0.4],
                               [0.2, 0.5],
                               [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])

    # 3층(출력층): 은닉(2) -> 출력(2)
    network['W3'] = np.array([[0.1, 0.3],
                               [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network


# 순전파
def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1   # 1층 가중합
    z1 = sigmoid(a1)           # 1층 활성화

    a2 = np.dot(z1, W2) + b2  # 2층 가중합
    z2 = sigmoid(a2)           # 2층 활성화

    a3 = np.dot(z2, W3) + b3  # 출력층 가중합
    y = identity_function(a3)  # 출력층 활성화 (항등 함수)

    return y


network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)
