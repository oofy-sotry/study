# coding: utf-8
"""8.2 딥러닝의 초기 역사 - (1) GoogLeNet 1x1 conv의 매개변수 절감 효과, (2) ResNet 스킵 연결의 기울기 소실 완화 효과"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False


# ---------------------------------------------------------------------------
# 8.2.3 GoogLeNet : 1x1 conv로 채널 수를 줄인 뒤 3x3 conv를 적용할 때의 매개변수 절감 비교
# ---------------------------------------------------------------------------
def direct_conv_params(c_in, c_out, kernel_size=3):
    """1x1 병목 없이 곧바로 c_in -> c_out 3x3 conv를 적용할 때의 매개변수 수"""
    return c_in * c_out * kernel_size * kernel_size


def bottleneck_conv_params(c_in, c_mid, c_out, kernel_size=3):
    """1x1 conv(c_in->c_mid)로 채널을 줄인 뒤 3x3 conv(c_mid->c_out)를 적용할 때의 매개변수 수"""
    reduce_1x1 = c_in * c_mid * 1 * 1
    conv_3x3 = c_mid * c_out * kernel_size * kernel_size
    return reduce_1x1 + conv_3x3


def print_inception_comparison():
    print('=== GoogLeNet 인셉션 구조 : 1x1 conv 병목(bottleneck) 유무에 따른 매개변수 비교 ===')
    print('(c_in : 입력 채널 수, c_mid : 1x1 conv로 줄인 채널 수, c_out : 3x3 conv 출력 채널 수)')
    print(f'{"c_in":>6} | {"c_mid":>6} | {"c_out":>6} | {"직접 3x3 conv":>14} | {"1x1+3x3 conv":>14} | 절감률')
    cases = [
        (192, 16, 32),
        (192, 32, 64),
        (480, 16, 32),
    ]
    for c_in, c_mid, c_out in cases:
        direct = direct_conv_params(c_in, c_out)
        bottleneck = bottleneck_conv_params(c_in, c_mid, c_out)
        saving = (1 - bottleneck / direct) * 100
        print(f'{c_in:>6} | {c_mid:>6} | {c_out:>6} | {direct:>14} | {bottleneck:>14} | {saving:5.1f}%')
    print('-> 1x1 conv로 채널 수를 먼저 줄이면(병목), 3x3 conv가 처리할 채널 수가 줄어 매개변수가 크게 절감됨')
    print('-> GoogLeNet은 이런 1x1 conv를 인셉션 구조 곳곳에 배치해 매개변수 제거와 고속화를 동시에 달성')


# ---------------------------------------------------------------------------
# 8.2.4 ResNet : 스킵 연결(skip connection)이 기울기 소실을 완화하는 효과 확인
# ---------------------------------------------------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = sigmoid(x)
        return self.out

    def backward(self, dout):
        return dout * self.out * (1 - self.out)


class Block:
    """Affine -> Sigmoid 로 구성된 한 개의 변환 F(x). plain/residual 두 네트워크가 이 블록을 공유해서 사용"""
    def __init__(self, width, rng):
        # 표준편차를 작게 잡아 층이 깊어질수록 기울기가 잘 사라지는 상황을 재현
        W = rng.randn(width, width) * 0.5
        b = np.zeros(width)
        self.affine = Affine(W, b)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        return self.sigmoid.forward(self.affine.forward(x))

    def backward(self, dout):
        return self.affine.backward(self.sigmoid.backward(dout))


class PlainDeepNet:
    """스킵 연결이 없는 일반 네트워크 : x_(l+1) = F_l(x_l)"""
    def __init__(self, width, num_layers, rng):
        self.blocks = [Block(width, rng) for _ in range(num_layers)]

    def forward(self, x):
        for block in self.blocks:
            x = block.forward(x)
        return x

    def backward(self, dout):
        for block in reversed(self.blocks):
            dout = block.backward(dout)
        return dout


class ResDeepNet:
    """스킵 연결이 있는 네트워크(ResNet 스타일) : x_(l+1) = x_l + F_l(x_l)"""
    def __init__(self, width, num_layers, rng):
        self.blocks = [Block(width, rng) for _ in range(num_layers)]

    def forward(self, x):
        for block in self.blocks:
            x = x + block.forward(x)
        return x

    def backward(self, dout):
        # y = x + F(x) 이므로 dy/dx = I + dF/dx -> dx = dout + F.backward(dout)
        for block in reversed(self.blocks):
            dout = dout + block.backward(dout)
        return dout


def gradient_norm_at_input(net, x):
    net.forward(x)
    dout = np.ones_like(x)  # 출력 쪽에서 흘러들어오는 기울기(단순화를 위해 전부 1로 가정)
    dx = net.backward(dout)
    return np.linalg.norm(dx)


def compare_vanishing_gradient(width=10, depths=(2, 5, 10, 20, 40, 80), seed=0):
    print()
    print('=== plain 네트워크 vs residual(스킵 연결) 네트워크 : 층 깊이에 따른 입력단 기울기 크기 비교 ===')
    print(f'{"층 수":>6} | {"plain(스킵無) 기울기 노름":>22} | {"residual(스킵有) 기울기 노름":>24}')
    plain_norms, res_norms = [], []
    for depth in depths:
        rng = np.random.RandomState(seed)
        x = rng.randn(1, width)

        plain_net = PlainDeepNet(width, depth, np.random.RandomState(seed + 1))
        res_net = ResDeepNet(width, depth, np.random.RandomState(seed + 1))

        plain_norm = gradient_norm_at_input(plain_net, x)
        res_norm = gradient_norm_at_input(res_net, x)
        plain_norms.append(plain_norm)
        res_norms.append(res_norm)
        print(f'{depth:>6} | {plain_norm:>22.6e} | {res_norm:>24.6e}')

    print('-> 층이 깊어질수록 plain 네트워크의 기울기 노름은 0에 가깝게 소실됨 (시그모이드 미분값<=0.25가 층마다 곱해짐)')
    print('-> residual 네트워크는 스킵 연결(항등 경로)이 기울기를 그대로 통과시켜, 층이 깊어져도 기울기가 크게 줄지 않음')
    return depths, plain_norms, res_norms


def plot_vanishing_gradient(depths, plain_norms, res_norms, save_path):
    plt.figure(figsize=(7, 5))
    plt.plot(depths, plain_norms, marker='o', label='plain (스킵 연결 없음)')
    plt.plot(depths, res_norms, marker='o', label='residual (스킵 연결 있음)')
    plt.yscale('log')
    plt.xlabel('네트워크 층 수')
    plt.ylabel('입력단 기울기 노름 (log scale)')
    plt.title('ResNet 스킵 연결의 기울기 소실 완화 효과')
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f'저장됨 : {save_path}')


if __name__ == '__main__':
    print_inception_comparison()

    depths, plain_norms, res_norms = compare_vanishing_gradient()
    save_path = os.path.join(os.path.dirname(__file__), 'vanishing_gradient.png')
    plot_vanishing_gradient(depths, plain_norms, res_norms, save_path)
