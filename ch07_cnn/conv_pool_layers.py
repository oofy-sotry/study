# coding: utf-8
"""7.4.3~7.4.4 합성곱/풀링 계층 구현하기 - Convolution, Pooling 클래스 (im2col 기반)"""
import numpy as np


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
        # 역전파용 중간 데이터
        self.x = None
        self.col = None
        self.col_W = None
        # 가중치·편향의 기울기
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


def numerical_gradient(f, x):
    """중심 차분으로 x와 형상이 같은 수치 기울기를 구함"""
    h = 1e-4
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp = x[idx]
        x[idx] = tmp + h
        fxh1 = f()
        x[idx] = tmp - h
        fxh2 = f()
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp
        it.iternext()
    return grad


if __name__ == '__main__':
    np.random.seed(0)

    # --- Convolution 순전파 : 출력 형상이 공식과 일치하는지 확인 ---
    print('=== Convolution 순전파 형상 확인 ===')
    x = np.random.rand(2, 1, 28, 28)          # MNIST 유사 배치
    W = np.random.randn(30, 1, 5, 5) * 0.01
    b = np.zeros(30)
    conv = Convolution(W, b, stride=1, pad=0)
    out = conv.forward(x)
    print(f'입력 {x.shape} -> Convolution -> 출력 {out.shape}  (공식 OH=(28-5)/1+1=24 와 일치)')

    # --- Convolution 기울기 확인 (수치 미분 vs 역전파) ---
    print()
    print('=== Convolution 기울기 확인 (수치 미분 vs 역전파) ===')
    x_small = np.random.rand(1, 1, 4, 4)
    W_small = np.random.randn(2, 1, 2, 2) * 0.5
    b_small = np.random.randn(2) * 0.5
    conv_small = Convolution(W_small, b_small, stride=1, pad=0)

    def loss():
        out = conv_small.forward(x_small)
        return np.sum(out ** 2)

    dx_numerical = numerical_gradient(lambda: loss(), x_small)
    dW_numerical = numerical_gradient(lambda: loss(), conv_small.W)
    db_numerical = numerical_gradient(lambda: loss(), conv_small.b)

    out = conv_small.forward(x_small)
    dout = 2 * out
    dx_backward = conv_small.backward(dout)

    for name, num, back in [('dx', dx_numerical, dx_backward),
                             ('dW', dW_numerical, conv_small.dW),
                             ('db', db_numerical, conv_small.db)]:
        diff = np.max(np.abs(num - back))
        print(f'{name} 최대 오차: {diff:.2e}  ->  {"통과" if diff < 1e-4 else "실패"}')

    # --- Pooling 순전파/역전파 확인 ---
    print()
    print('=== Pooling 순전파 형상 및 기울기 확인 ===')
    x_pool = np.random.rand(1, 2, 4, 4)
    pool = Pooling(pool_h=2, pool_w=2, stride=2)
    out_pool = pool.forward(x_pool)
    print(f'입력 {x_pool.shape} -> Pooling(2,2,stride=2) -> 출력 {out_pool.shape}')

    def pool_loss():
        out = pool.forward(x_pool)
        return np.sum(out ** 2)

    dx_pool_numerical = numerical_gradient(lambda: pool_loss(), x_pool)
    out_pool = pool.forward(x_pool)
    dx_pool_backward = pool.backward(2 * out_pool)
    diff_pool = np.max(np.abs(dx_pool_numerical - dx_pool_backward))
    print(f'Pooling dx 최대 오차: {diff_pool:.2e}  ->  {"통과" if diff_pool < 1e-4 else "실패"}')

    # --- Conv-ReLU-Pool 파이프라인으로 shape 흐름 확인 (7.1에서 예고한 흐름) ---
    print()
    print('=== Conv-ReLU-Pool 파이프라인 shape 흐름 (7.1에서 예고한 CNN 계층 구성) ===')
    x_pipe = np.random.rand(2, 1, 28, 28)
    conv1 = Convolution(np.random.randn(30, 1, 5, 5) * 0.01, np.zeros(30), stride=1, pad=0)
    relu1 = Relu()
    pool1 = Pooling(pool_h=2, pool_w=2, stride=2)

    h = conv1.forward(x_pipe)
    print(f'Conv1  : {x_pipe.shape} -> {h.shape}')
    h = relu1.forward(h)
    print(f'Relu1  : {h.shape} -> {h.shape} (형상 변화 없음)')
    h = pool1.forward(h)
    print(f'Pool1  : -> {h.shape}')
