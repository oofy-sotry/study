# coding: utf-8
"""7.1 전체 구조 - 완전연결(Affine) 신경망 vs CNN 계층 구성 비교"""
import numpy as np
from collections import OrderedDict


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        return dout


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        return (self.y - self.t) / batch_size


class FullyConnectedNet:
    """지금까지 다룬 완전연결(Affine) 신경망 - "Affine-ReLU"를 반복하고 마지막은 "Affine-Softmax" """

    def __init__(self, input_size, hidden_size_list, output_size):
        sizes = [input_size] + hidden_size_list + [output_size]
        self.params = {}
        for i in range(len(sizes) - 1):
            self.params[f'W{i+1}'] = 0.01 * np.random.randn(sizes[i], sizes[i + 1])
            self.params[f'b{i+1}'] = np.zeros(sizes[i + 1])

        self.layers = OrderedDict()
        for i in range(len(sizes) - 1):
            self.layers[f'Affine{i+1}'] = Affine(self.params[f'W{i+1}'], self.params[f'b{i+1}'])
            if i < len(sizes) - 2:
                self.layers[f'Relu{i+1}'] = Relu()

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def layer_flow(self):
        return list(self.layers.keys()) + ['SoftmaxWithLoss']


def cnn_layer_flow(num_conv_blocks=3):
    """CNN 계층 구성 - Convolution/Pooling 계층은 7.4~7.5에서 구현하므로 여기서는 흐름만 나열"""
    flow = []
    for i in range(num_conv_blocks):
        flow += [f'Conv{i+1}', f'Relu{i+1}', f'Pool{i+1}']
    flow += ['Affine1', f'Relu{num_conv_blocks+1}', 'Affine2', 'SoftmaxWithLoss']
    return flow


if __name__ == '__main__':
    print('=== 완전연결(Affine) 신경망 계층 구성 ===')
    fc_net = FullyConnectedNet(input_size=784, hidden_size_list=[100, 100], output_size=10)
    print(' -> '.join(fc_net.layer_flow()))

    x = np.random.rand(2, 784)
    t = np.zeros((2, 10))
    t[np.arange(2), np.random.randint(0, 10, 2)] = 1
    loss = fc_net.loss(x, t)
    print(f'더미 데이터로 순전파한 손실값: {loss:.4f}')

    print()
    print('=== CNN 계층 구성 (Convolution/Pooling은 7.4~7.5에서 구현 예정) ===')
    print(' -> '.join(cnn_layer_flow()))

    print()
    print('공통점: 출력에 가까운 층은 Affine-ReLU, 마지막 층은 Affine-Softmax 그대로 사용')
    print('차이점: CNN은 "Convolution-ReLU-(Pooling)" 흐름의 계층이 새로 추가됨')
