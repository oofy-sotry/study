# coding: utf-8
"""7.2 합성곱 계층 - 패딩/스트라이드를 반영한 단순(loop) 합성곱 연산 구현
(im2col을 이용한 고속화는 7.4에서 다룸. 여기서는 정의 그대로 4중 for문으로 구현)
"""
import numpy as np


def conv_output_size(input_size, filter_size, stride=1, pad=0):
    """출력 크기 계산 공식 : OH = (H + 2P - FH) / S + 1"""
    return (input_size + 2 * pad - filter_size) // stride + 1


def pad_input(x, pad):
    """x: (N, C, H, W) 4차원 데이터의 상하좌우에 0으로 패딩"""
    if pad == 0:
        return x
    return np.pad(x, [(0, 0), (0, 0), (pad, pad), (pad, pad)], mode='constant')


def convolution(x, W, b, stride=1, pad=0):
    """단순 합성곱 연산 (배치 처리 + 다중 채널 + 다중 필터 지원)

    Parameters
    ----------
    x : 입력 데이터, 형상 (N, C, H, W)
    W : 필터(가중치), 형상 (FN, C, FH, FW)
    b : 편향, 형상 (FN,)
    stride, pad : 스트라이드, 패딩 폭

    Returns
    -------
    출력 데이터, 형상 (N, FN, OH, OW)
    """
    N, C, H, W_in = x.shape
    FN, FC, FH, FW = W.shape
    assert C == FC, f'입력 채널({C})과 필터 채널({FC})이 일치해야 함'

    OH = conv_output_size(H, FH, stride, pad)
    OW = conv_output_size(W_in, FW, stride, pad)

    x_padded = pad_input(x, pad)
    out = np.zeros((N, FN, OH, OW))

    for n in range(N):                      # 배치 데이터마다
        for f in range(FN):                 # 필터(출력 채널)마다
            for i in range(OH):
                for j in range(OW):
                    h_start, w_start = i * stride, j * stride
                    window = x_padded[n, :, h_start:h_start + FH, w_start:w_start + FW]
                    # 채널별 합성곱 결과를 더해서 하나의 출력으로 (단일 곱셈-누산)
                    out[n, f, i, j] = np.sum(window * W[f]) + b[f]

    return out


if __name__ == '__main__':
    # --- 출력 크기 공식 확인 : 입력(4,4) + 필터(3,3), 패딩 0, 스트라이드 1 → 출력(2,2) ---
    print('=== 출력 크기 공식 확인 ===')
    print('입력(4,4) + 필터(3,3), pad=0, stride=1 ->', conv_output_size(4, 3, stride=1, pad=0))
    print('입력(4,4) + 필터(3,3), pad=1, stride=1 ->', conv_output_size(4, 3, stride=1, pad=1))
    print('입력(7,7) + 필터(3,3), pad=0, stride=2 ->', conv_output_size(7, 3, stride=2, pad=0))

    # --- 단일 채널, 단일 필터로 기본 동작 확인 ---
    print()
    print('=== 단일 채널 합성곱 (패딩 없음) ===')
    x = np.arange(16).reshape(1, 1, 4, 4).astype(float)
    W = np.array([[[[1, 0], [0, 1]]]], dtype=float)  # (FN=1, C=1, FH=2, FW=2)
    b = np.array([0.0])
    out = convolution(x, W, b, stride=1, pad=0)
    print('입력 형상:', x.shape, '-> 출력 형상:', out.shape)
    print(out[0, 0])

    # --- 패딩을 적용하면 출력 크기가 입력과 같아짐(same padding 예) ---
    print()
    print('=== 패딩 적용 (pad=1) : 입력과 같은 크기의 출력 ===')
    out_padded = convolution(x, W, b, stride=1, pad=1)
    print('입력 형상:', x.shape, '-> 출력 형상:', out_padded.shape)

    # --- 3차원 데이터(다중 채널) + 다중 필터 + 배치 처리 ---
    print()
    print('=== 다중 채널 + 다중 필터 + 배치 처리 (MNIST 유사 형상) ===')
    N, C, H, W_size = 2, 1, 28, 28
    FN, FH, FW = 30, 5, 5
    x_batch = np.random.rand(N, C, H, W_size)
    W_filters = np.random.randn(FN, C, FH, FW) * 0.01
    b_filters = np.zeros(FN)
    out_batch = convolution(x_batch, W_filters, b_filters, stride=1, pad=0)
    print(f'입력 형상 (N,C,H,W): {x_batch.shape}')
    print(f'필터 형상 (FN,C,FH,FW): {W_filters.shape}')
    print(f'출력 형상 (N,FN,OH,OW): {out_batch.shape}')
    expected_oh = conv_output_size(H, FH, stride=1, pad=0)
    print(f'공식으로 계산한 OH/OW: {expected_oh} -> 실제 출력과 일치: {out_batch.shape[2] == expected_oh}')
