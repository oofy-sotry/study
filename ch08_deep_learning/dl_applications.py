# coding: utf-8
"""8.4 딥러닝의 활용 - (1) 사물 검출 : IoU/NMS, (2) 분할 : FCN 방식 vs 픽셀별 방식 속도 비교 + 이중선형보간 확대, (3) 사진 캡션 생성 : CNN 특징 + RNN 미니 NIC"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False


# ---------------------------------------------------------------------------
# 8.4.1 사물 검출 : IoU(Intersection over Union) 계산 + NMS(Non-Maximum Suppression)
#   R-CNN 계열은 후보 영역(region proposal)을 다수 만들어낸 뒤, 같은 사물을 가리키는
#   중복 박스를 걸러내야 함 -> 이때 쓰이는 핵심 후처리 알고리즘이 IoU 기반 NMS
# ---------------------------------------------------------------------------
def iou(box_a, box_b):
    """box = (x1, y1, x2, y2). 두 박스의 교집합 넓이 / 합집합 넓이"""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def nms(boxes, scores, iou_threshold=0.5):
    """점수가 높은 박스부터 채택하고, 이미 채택된 박스와 IoU가 임계값을 넘는 박스는 제거(같은 사물의 중복 검출로 간주)"""
    order = np.argsort(scores)[::-1]  # 점수 내림차순 인덱스
    keep = []
    while len(order) > 0:
        current = order[0]
        keep.append(current)
        rest = order[1:]
        remaining = [i for i in rest if iou(boxes[current], boxes[i]) < iou_threshold]
        order = np.array(remaining, dtype=int)
    return keep


def demo_object_detection(save_path):
    # 같은 사물(사람) 하나를 놓고 R-CNN이 만들어낸 여러 후보 영역이라고 가정 (서로 겹치는 박스들 + 별개의 사물 박스 1개)
    boxes = np.array([
        [50, 50, 150, 200],   # 사물 A 후보 1
        [55, 45, 145, 205],   # 사물 A 후보 2 (거의 동일 위치)
        [60, 60, 140, 190],   # 사물 A 후보 3 (약간 작음)
        [220, 80, 300, 220],  # 사물 B (별개의 사물)
    ], dtype=float)
    scores = np.array([0.75, 0.9, 0.6, 0.8])

    keep = nms(boxes, scores, iou_threshold=0.5)

    print('=== 사물 검출 후처리 : IoU 기반 NMS(Non-Maximum Suppression) ===')
    for i, (box, score) in enumerate(zip(boxes, scores)):
        mark = 'KEEP' if i in keep else 'removed'
        print(f'box{i} = {box.tolist()}, score={score:.2f} -> {mark}')
    print(f'-> 겹치는(IoU>=0.5) 박스 중 점수가 가장 높은 박스만 남기고 나머지는 제거')
    print(f'-> 원래 후보 {len(boxes)}개 -> NMS 이후 {len(keep)}개 (사물마다 박스 1개로 정리됨)')

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, indices, title in zip(axes, [range(len(boxes)), keep], ['NMS 적용 전 (후보 영역 전부)', 'NMS 적용 후 (중복 제거)']):
        ax.set_xlim(0, 350)
        ax.set_ylim(250, 0)
        ax.set_title(title)
        for i in indices:
            x1, y1, x2, y2 = boxes[i]
            color = 'tab:red' if i in keep else 'tab:gray'
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, f'{scores[i]:.2f}', color=color, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f'저장됨 : {save_path}')


# ---------------------------------------------------------------------------
# 8.4.2 분할(Segmentation) : "픽셀 하나씩 forward" 방식 vs FCN 방식(슬라이딩 결과를 한 번에 재사용) 속도 비교
#   + 마지막에 공간 크기를 늘리는 이중 선형 보간(bilinear interpolation) 구현
# ---------------------------------------------------------------------------
def classify_patch(patch, weight):
    """3x3 패치 -> 점수 하나 (합성곱 필터 1개를 적용하는 것과 동일한 연산)"""
    return float(np.dot(patch.flatten(), weight))


def segment_pixel_by_pixel(image, weight):
    """가장 단순한(낭비가 많은) 방식 : 모든 픽셀에 대해 매번 패치를 새로 잘라 forward를 반복"""
    H, W = image.shape
    padded = np.pad(image, 1)
    out = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            patch = padded[i:i + 3, j:j + 3]
            out[i, j] = classify_patch(patch, weight)
    return out


def im2col_patches(image, ksize=3):
    """이미지의 모든 3x3 패치를 한 번에 잘라내 (H*W, ksize*ksize) 행렬로 반환 (7.4절 im2col과 동일한 원리)"""
    H, W = image.shape
    padded = np.pad(image, 1)
    col = np.zeros((H, W, ksize, ksize))
    for y in range(ksize):
        y_max = y + H
        for x in range(ksize):
            x_max = x + W
            col[:, :, y, x] = padded[y:y_max, x:x_max]
    return col.reshape(H * W, ksize * ksize)


def segment_fcn_style(image, weight):
    """FCN 방식 : im2col로 슬라이딩 패치를 한 번에 뽑아, 단일 행렬곱으로 전체 픽셀을 동시에 분류"""
    H, W = image.shape
    patches_mat = im2col_patches(image, ksize=3)
    scores = patches_mat @ weight  # (H*W,)
    return scores.reshape(H, W)


def bilinear_resize(img, new_h, new_w):
    """이중 선형 보간(bilinear interpolation)으로 img를 (new_h, new_w) 크기로 확대/축소"""
    H, W = img.shape
    out = np.zeros((new_h, new_w))
    row_scale = (H - 1) / (new_h - 1) if new_h > 1 else 0
    col_scale = (W - 1) / (new_w - 1) if new_w > 1 else 0

    for i in range(new_h):
        src_y = i * row_scale
        y0 = int(np.floor(src_y))
        y1 = min(y0 + 1, H - 1)
        dy = src_y - y0
        for j in range(new_w):
            src_x = j * col_scale
            x0 = int(np.floor(src_x))
            x1 = min(x0 + 1, W - 1)
            dx = src_x - x0

            top = img[y0, x0] * (1 - dx) + img[y0, x1] * dx
            bottom = img[y1, x0] * (1 - dx) + img[y1, x1] * dx
            out[i, j] = top * (1 - dy) + bottom * dy
    return out


def demo_segmentation(save_path, size=40, seed=0):
    rng = np.random.RandomState(seed)
    image = rng.rand(size, size)
    weight = rng.randn(9) * 0.3  # 3x3 필터 1개(=클래스 판별용 가중치)

    start = time.time()
    out_pixel = segment_pixel_by_pixel(image, weight)
    pixel_time = time.time() - start

    start = time.time()
    out_fcn = segment_fcn_style(image, weight)
    fcn_time = time.time() - start

    assert np.allclose(out_pixel, out_fcn, atol=1e-8), '두 방식의 픽셀별 분류 점수는 동일해야 함'

    print()
    print('=== 분할(Segmentation) : 픽셀별 forward 반복 vs FCN 방식(슬라이딩 결과 재사용) 속도 비교 ===')
    print(f'이미지 크기 {size}x{size} ({size * size}개 픽셀)')
    print(f'픽셀별로 매번 forward 반복 : {pixel_time:.4f}초')
    print(f'FCN 방식(한 번에 행렬곱)   : {fcn_time:.4f}초')
    print(f'속도 향상 배율             : {pixel_time / fcn_time:.1f}배')
    print('-> 결과(각 픽셀의 분류 점수)는 완전히 동일하지만, 슬라이딩 패치를 한 번에 모아 큰 행렬곱 1번으로')
    print('   처리하는 FCN 방식이 픽셀마다 forward를 새로 되풀이하는 방식보다 훨씬 빠름')

    # 저해상도로 분할한 결과를 원래 이미지 크기로 되돌리는 "이중 선형 보간(deconvolution의 역할)"
    low_res = out_fcn[::4, ::4]  # 1/4 크기로 다운샘플링한 저해상도 분할 맵이라고 가정
    upsampled = bilinear_resize(low_res, size, size)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('원본 이미지')
    axes[1].imshow(low_res, cmap='viridis')
    axes[1].set_title(f'저해상도 분할 맵 ({low_res.shape[0]}x{low_res.shape[1]})')
    axes[2].imshow(upsampled, cmap='viridis')
    axes[2].set_title(f'이중 선형 보간으로 확대 ({size}x{size})')
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f'저장됨 : {save_path}')


# ---------------------------------------------------------------------------
# 8.4.3 사진 캡션 생성 : CNN 특징 벡터를 RNN의 초기 은닉 상태로 사용하는 미니 NIC(Neural Image Caption) 구조 재현
#   (가중치는 학습되지 않은 무작위 값 - "문장 품질"이 아니라 "CNN 특징 -> RNN 순환 생성"이라는 데이터 흐름 구조를 확인하는 것이 목적)
# ---------------------------------------------------------------------------
VOCAB = ['<start>', '<end>', 'a', 'dog', 'cat', 'is', 'running', 'sleeping', 'on', 'grass']


class SimpleRNNCell:
    def __init__(self, feature_size, hidden_size, vocab_size, rng):
        self.Wx = rng.randn(vocab_size, hidden_size) * 0.1   # 단어 임베딩 -> 은닉 상태로의 가중치
        self.Wh = rng.randn(hidden_size, hidden_size) * 0.1  # 은닉 상태 -> 은닉 상태 가중치
        self.Wout = rng.randn(hidden_size, vocab_size) * 0.1  # 은닉 상태 -> 단어별 점수
        self.b = np.zeros(hidden_size)

    def step(self, word_onehot, h_prev):
        h_next = np.tanh(np.dot(word_onehot, self.Wx) + np.dot(h_prev, self.Wh) + self.b)
        scores = np.dot(h_next, self.Wout)
        return h_next, scores


def one_hot(index, size):
    vec = np.zeros(size)
    vec[index] = 1.0
    return vec


def generate_caption(cnn_feature, rnn, max_len=8):
    """NIC 구조 : CNN 특징 벡터를 RNN의 초기 은닉 상태로 사용해, <start> 토큰부터 순환적으로 다음 단어를 생성"""
    vocab_size = len(VOCAB)
    h = cnn_feature  # CNN이 추출한 사진 특징 벡터를 그대로 초기 은닉 상태 h0로 사용
    word_idx = VOCAB.index('<start>')
    caption = []

    for _ in range(max_len):
        word_vec = one_hot(word_idx, vocab_size)
        h, scores = rnn.step(word_vec, h)
        word_idx = int(np.argmax(scores))  # 가장 점수가 높은 단어를 다음 단어로 선택(greedy decoding)
        if VOCAB[word_idx] == '<end>':
            break
        caption.append(VOCAB[word_idx])
    return caption


def demo_image_captioning(hidden_size=10, seed=0):
    rng = np.random.RandomState(seed)
    vocab_size = len(VOCAB)
    rnn = SimpleRNNCell(feature_size=hidden_size, hidden_size=hidden_size, vocab_size=vocab_size, rng=rng)

    print()
    print('=== 사진 캡션 생성 : CNN 특징 벡터 -> RNN 초기 은닉 상태 -> 순환적 단어 생성 (미니 NIC) ===')
    print('(가중치가 학습되지 않은 무작위 값이므로 생성 문장 자체는 의미가 없음 - 구조/데이터 흐름 확인용)')
    for photo_id in range(3):
        # 서로 다른 사진마다 CNN이 뽑아냈다고 가정한 서로 다른 특징 벡터
        cnn_feature = rng.randn(hidden_size) * 0.5
        caption = generate_caption(cnn_feature, rnn)
        print(f'사진 {photo_id} (CNN 특징 벡터로 조건화) -> 생성된 캡션 : {" ".join(caption)}')
    print('-> 같은 RNN(가중치 동일)이라도 CNN에서 넘어온 초기 은닉 상태(사진 특징)가 다르면 서로 다른 문장이 생성됨')
    print('-> 이렇게 이미지(CNN)와 텍스트(RNN)라는 서로 다른 종류의 정보를 조합하는 것이 멀티모달(multimodal) 처리')


if __name__ == '__main__':
    demo_object_detection(os.path.join(os.path.dirname(__file__), 'nms_detection.png'))
    demo_segmentation(os.path.join(os.path.dirname(__file__), 'segmentation_upsampling.png'))
    demo_image_captioning()
