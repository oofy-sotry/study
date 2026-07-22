# coding: utf-8
"""8.1 더 깊게 - (1) 작은 필터를 쌓을 때의 매개변수 절감 효과, (2) 데이터 확장(augmentation) 구현"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
from dataset.mnist import load_mnist


# ---------------------------------------------------------------------------
# 8.1.3 깊게 하는 이유 (1) : 작은 필터(3x3)를 쌓았을 때의 수용 영역·매개변수 비교
# ---------------------------------------------------------------------------
def receptive_field(num_layers, kernel_size=3):
    """3x3 합성곱을 num_layers번 쌓았을 때의 수용 영역(receptive field) 크기"""
    return 1 + num_layers * (kernel_size - 1)


def stacked_params(num_layers, kernel_size=3):
    """3x3 합성곱을 num_layers번 쌓았을 때 필요한 매개변수 수 (입출력 채널 1개 기준)"""
    return num_layers * kernel_size * kernel_size


def single_params(equiv_size):
    """수용 영역과 동일한 크기의 필터 1장을 한 번에 쓸 때 필요한 매개변수 수"""
    return equiv_size * equiv_size


def print_depth_comparison():
    print('=== 3x3 합성곱을 쌓을 때 vs 동일 크기의 필터 1장을 쓸 때 ===')
    print('(매개변수 수는 입력 채널 1개 -> 출력 채널 1개 기준 순수 필터 값 개수)')
    print(f'{"3x3 층 수":>8} | {"수용 영역":>8} | {"쌓았을 때 params":>16} | {"단일 필터 params":>16} | 절감률')
    for n in (1, 2, 3, 4):
        rf = receptive_field(n)
        p_stacked = stacked_params(n)
        p_single = single_params(rf)
        saving = (1 - p_stacked / p_single) * 100
        print(f'{n:>8} | {rf:>6}x{rf:<1} | {p_stacked:>16} | {p_single:>16} | {saving:5.1f}%')
    print('-> 층을 깊게(작은 필터를 여러 번) 쌓을수록 같은 수용 영역을 더 적은 매개변수로 커버')
    print('-> 게다가 층마다 ReLU가 끼어들어 비선형성이 늘어나므로 표현력은 매개변수 수 이상으로 증가')


# ---------------------------------------------------------------------------
# 8.1.2 정확도를 더 높이려면 : 데이터 확장(Data Augmentation)
# ---------------------------------------------------------------------------
def shift(img, dy, dx):
    """이미지를 (dy, dx)만큼 평행 이동, 빈 공간은 0으로 채움"""
    H, W = img.shape
    shifted = np.zeros_like(img)
    src_y0, src_y1 = max(0, -dy), min(H, H - dy)
    dst_y0, dst_y1 = max(0, dy), min(H, H + dy)
    src_x0, src_x1 = max(0, -dx), min(W, W - dx)
    dst_x0, dst_x1 = max(0, dx), min(W, W + dx)
    shifted[dst_y0:dst_y1, dst_x0:dst_x1] = img[src_y0:src_y1, src_x0:src_x1]
    return shifted


def zoom(img, scale):
    """중심 기준 확대(scale>1)/축소(scale<1), 최근접 이웃 방식으로 원래 크기 유지"""
    H, W = img.shape
    new_H, new_W = max(1, int(round(H * scale))), max(1, int(round(W * scale)))
    row_idx = np.clip((np.arange(new_H) / scale).astype(int), 0, H - 1)
    col_idx = np.clip((np.arange(new_W) / scale).astype(int), 0, W - 1)
    resized = img[row_idx][:, col_idx]

    out = np.zeros((H, W))
    if scale >= 1:
        y0, x0 = (new_H - H) // 2, (new_W - W) // 2
        out = resized[y0:y0 + H, x0:x0 + W]
    else:
        y0, x0 = (H - new_H) // 2, (W - new_W) // 2
        out[y0:y0 + new_H, x0:x0 + new_W] = resized
    return out


def adjust_brightness(img, factor):
    """밝기를 factor배로 조정 (0~1 범위로 클리핑)"""
    return np.clip(img * factor, 0, 1)


def random_crop(img, crop_ratio=0.75, rng=None):
    """무작위 위치에서 crop_ratio 비율만큼 잘라낸 뒤, 원래 크기로 0-패딩"""
    rng = rng or np.random
    H, W = img.shape
    ch, cw = int(H * crop_ratio), int(W * crop_ratio)
    y0 = rng.randint(0, H - ch + 1)
    x0 = rng.randint(0, W - cw + 1)
    cropped = img[y0:y0 + ch, x0:x0 + cw]

    out = np.zeros((H, W))
    py0, px0 = (H - ch) // 2, (W - cw) // 2
    out[py0:py0 + ch, px0:px0 + cw] = cropped
    return out


def augment_demo(image, save_path, seed=0):
    rng = np.random.RandomState(seed)
    variants = [
        ('원본', image),
        ('이동(위로 4px)', shift(image, -4, 0)),
        ('이동(오른쪽 4px)', shift(image, 0, 4)),
        ('확대(1.3배)', zoom(image, 1.3)),
        ('축소(0.7배)', zoom(image, 0.7)),
        ('밝기 증가(x1.5)', adjust_brightness(image, 1.5)),
        ('밝기 감소(x0.5)', adjust_brightness(image, 0.5)),
        ('랜덤 crop(75%)', random_crop(image, 0.75, rng)),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(10, 6))
    fig.suptitle('데이터 확장(Data Augmentation) 예시 - 숫자 인식은 flip(좌우반전) 미사용')
    for ax, (name, img) in zip(axes.flat, variants):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(name, fontsize=9)
        ax.axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(hspace=0.4)
    plt.savefig(save_path)
    plt.close(fig)
    print(f'저장됨 : {save_path}')


if __name__ == '__main__':
    print_depth_comparison()

    print()
    print('=== 데이터 확장(Data Augmentation) 시각화 ===')
    x_train, t_train, x_test, t_test = load_mnist(normalize=True, flatten=False, one_hot_label=True)
    sample = x_train[0, 0]  # (28, 28), 정규화된 값(0~1)
    save_path = os.path.join(os.path.dirname(__file__), 'data_augmentation.png')
    augment_demo(sample, save_path)
