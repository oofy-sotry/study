# coding: utf-8
"""7.4.1~7.4.2 im2col / col2im - 4차원 데이터를 2차원 행렬로 전개(복원)"""
import numpy as np


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """입력 데이터를 필터 적용 영역 기준으로 2차원 행렬로 전개

    Parameters
    ----------
    input_data : (N, C, H, W) 형상의 4차원 입력 데이터
    filter_h, filter_w : 필터의 높이, 너비
    stride, pad : 스트라이드, 패딩 폭

    Returns
    -------
    col : (N*out_h*out_w, C*filter_h*filter_w) 형상의 2차원 배열
    """
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
    """im2col로 전개한 2차원 행렬을 (N, C, H, W) 형상으로 복원 (역전파용)

    겹치는 영역은 값을 더해서 복원함 (im2col의 정확한 역변환이 아니라
    합성곱/풀링의 역전파에서 기울기를 누적하는 용도)
    """
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


if __name__ == '__main__':
    # --- im2col 출력 형상 확인 ---
    print('=== im2col 출력 형상 확인 ===')
    x1 = np.random.rand(1, 3, 7, 7)   # 데이터 1개, 채널 3, 7x7
    col1 = im2col(x1, filter_h=5, filter_w=5, stride=1, pad=0)
    print(f'입력 {x1.shape} + 필터(5,5), stride=1, pad=0 -> col 형상 {col1.shape}')
    print('  (out_h=out_w=3 이므로 행 수 = 1*3*3=9, 열 수 = C*FH*FW = 3*5*5=75)')

    x2 = np.random.rand(10, 3, 7, 7)  # 배치 10개
    col2 = im2col(x2, filter_h=5, filter_w=5, stride=1, pad=0)
    print(f'입력 {x2.shape} + 필터(5,5), stride=1, pad=0 -> col 형상 {col2.shape}')

    # --- im2col + 행렬곱으로 7.2의 수동 계산 결과 재현 ---
    print()
    print('=== im2col + np.dot 으로 합성곱 재현 (7.2 수동 계산과 비교) ===')
    x = np.arange(16).reshape(1, 1, 4, 4).astype(float)
    W = np.array([[[[1, 0], [0, 1]]]], dtype=float)  # (FN=1, C=1, FH=2, FW=2)
    b = np.array([0.0])

    FN, C, FH, FW = W.shape
    N, C_, H, W_size = x.shape
    out_h = (H - FH) // 1 + 1
    out_w = (W_size - FW) // 1 + 1

    col = im2col(x, FH, FW, stride=1, pad=0)
    col_W = W.reshape(FN, -1).T
    out = np.dot(col, col_W) + b
    out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

    print('im2col 기반 결과:')
    print(out[0, 0])
    print('7.2 수동(4중 for문) 결과: [[5,7,9],[13,15,17],[21,23,25]] 와 일치 여부:',
          np.array_equal(out[0, 0], np.array([[5., 7., 9.], [13., 15., 17.], [21., 23., 25.]])))

    # --- col2im 은 im2col의 역전파(기울기 누적)용 함수임을 확인 ---
    print()
    print('=== col2im : im2col 결과를 원래 입력 형상으로 복원(누적) ===')
    restored = col2im(col, x.shape, FH, FW, stride=1, pad=0)
    print(f'col {col.shape} -> col2im -> {restored.shape} (입력 형상 {x.shape} 과 동일)')
