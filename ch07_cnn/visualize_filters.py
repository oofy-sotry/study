# coding: utf-8
"""7.6 CNN 시각화하기 - 1번째 층 필터를 학습 전/후로 시각화하고, 학습된 필터가
만드는 특징 맵(feature map)을 확인한다.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from collections import OrderedDict
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 화면 없이 파일로만 저장
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'AppleGothic'  # 한글 폰트 (macOS)
plt.rcParams['axes.unicode_minus'] = False
from dataset.mnist import load_mnist


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], mode='constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        self.x = None
        self.col = None
        self.col_W = None
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2 * self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2 * self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T
        out = np.dot(col, col_W) + self.b

        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x, self.col, self.col_W = x, col, col_W
        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout).transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)
        return dx


class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x, self.arg_max = x, arg_max
        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
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


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx.reshape(*self.original_x_shape)


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


class SimpleConvNet:
    def __init__(self, input_dim=(1, 28, 28),
                 conv_param={'filter_num': 16, 'filter_size': 5, 'pad': 0, 'stride': 1},
                 hidden_size=50, output_size=10, weight_init_std=0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2 * filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size / 2))

        self.params = {}
        self.params['W1'] = weight_init_std * \
            np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * \
            np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * \
            np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                            conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def gradient(self, x, t):
        self.loss(x, t)
        dout = self.last_layer.backward(1)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W3'], grads['b3'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        return grads


def plot_filters(filters, title, save_path):
    """filters : (FN, 1, FH, FW) 형상의 필터를 그리드로 저장"""
    FN = filters.shape[0]
    cols = 4
    rows = int(np.ceil(FN / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    fig.suptitle(title)
    for i, ax in enumerate(axes.flat):
        ax.axis('off')
        if i < FN:
            ax.imshow(filters[i, 0], cmap='gray', interpolation='nearest')
            ax.set_title(f'#{i}', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f'저장됨 : {save_path}')


def plot_feature_maps(image, feature_maps, save_path):
    """image : (1, H, W) 원본 이미지, feature_maps : (FN, OH, OW) Conv1+ReLU1 출력"""
    FN = feature_maps.shape[0]
    cols = 4
    rows = int(np.ceil((FN + 1) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    fig.suptitle('원본 이미지 vs 1번째 합성곱 계층 특징 맵(feature map)')
    for i, ax in enumerate(axes.flat):
        ax.axis('off')
        if i == 0:
            ax.imshow(image[0], cmap='gray')
            ax.set_title('input', fontsize=8)
        elif i - 1 < FN:
            ax.imshow(feature_maps[i - 1], cmap='gray', interpolation='nearest')
            ax.set_title(f'filter #{i-1}', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f'저장됨 : {save_path}')


if __name__ == '__main__':
    np.random.seed(0)
    save_dir = os.path.dirname(__file__)

    network = SimpleConvNet(
        input_dim=(1, 28, 28),
        conv_param={'filter_num': 16, 'filter_size': 5, 'pad': 0, 'stride': 1},
        hidden_size=50, output_size=10, weight_init_std=0.01)

    # --- 학습 전 필터 저장 (무작위 초기화 상태) ---
    W1_before = network.params['W1'].copy()
    plot_filters(W1_before, '학습 전 1번째 층 필터 (무작위 초기화)',
                 os.path.join(save_dir, 'filters_before.png'))

    # --- MNIST 일부 데이터로 학습 (7.5와 동일한 설정) ---
    print()
    print('=== MNIST 일부 데이터로 학습 ===')
    x_train, t_train, x_test, t_test = load_mnist(normalize=True, flatten=False, one_hot_label=True)
    train_size = 300
    x_train, t_train = x_train[:train_size], t_train[:train_size]

    iters_num = 300
    batch_size = 30
    learning_rate = 0.1
    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch, t_batch = x_train[batch_mask], t_train[batch_mask]
        grad = network.gradient(x_batch, t_batch)
        for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'):
            network.params[key] -= learning_rate * grad[key]
        if i % 100 == 0 or i == iters_num - 1:
            print(f'iter {i:3d} : loss = {network.loss(x_batch, t_batch):.4f}')

    # --- 학습 후 필터 저장 ---
    W1_after = network.params['W1']
    plot_filters(W1_after, '학습 후 1번째 층 필터 (에지/블롭에 반응하도록 갱신됨)',
                 os.path.join(save_dir, 'filters_after.png'))

    # --- 학습 전/후 필터 값의 변화량으로 "규칙성이 생겼는지" 정량 확인 ---
    print()
    print('=== 학습 전/후 필터 변화 ===')
    diff = np.abs(W1_after - W1_before)
    print(f'필터 값 평균 변화량 : {diff.mean():.5f} (0이 아니면 필터가 실제로 갱신되었다는 뜻)')

    # --- 학습된 1번째 층 필터가 실제 숫자 이미지에서 무엇을 뽑아내는지 확인 ---
    print()
    print('=== 학습된 필터의 특징 맵(feature map) 확인 ===')
    sample_image = x_test[0]  # (1, 28, 28)
    conv1 = network.layers['Conv1']
    relu1 = network.layers['Relu1']
    feature_maps = relu1.forward(conv1.forward(sample_image[np.newaxis, ...]))[0]  # (FN, OH, OW)
    plot_feature_maps(sample_image, feature_maps, os.path.join(save_dir, 'feature_maps.png'))

    print()
    print('참고 : 이 SimpleConvNet은 합성곱 계층이 1개뿐이라 7.6.2의 "층이 깊어질수록')
    print('추상화된 정보를 추출한다"는 현상은 재현할 수 없음 (LeNet/AlexNet처럼')
    print('여러 합성곱 계층을 쌓아야 텍스처·사물 부분 단위의 고급 정보가 나타남)')
