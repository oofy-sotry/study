# coding: utf-8
"""8.3 더 빠르게(속도 개선) - (1) 벡터화(GPU 병렬성 축소판) 속도 비교, (2) 분산 학습(데이터 병렬) 기울기 평균 검증, (3) 연산 정밀도별 메모리·오차 비교"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False


# ---------------------------------------------------------------------------
# 8.3.2 GPU를 활용한 고속화 : "병렬(벡터화) 연산 vs 순차(for문) 연산"으로 병렬 처리의 위력을 축소 재현
#   실제 GPU는 없지만, CPU에서도 벡터화된 배치 연산이 파이썬 for문 순차 연산보다 훨씬 빠름을 확인해
#   "대량의 단순 곱셈-누적을 병렬로 처리하면 빨라진다"는 8.3.2의 핵심 아이디어를 체감
# ---------------------------------------------------------------------------
def sequential_matmul(X, W):
    """배치 샘플을 한 장씩 순차적으로(for문) 처리 - CPU의 '연속적 계산' 방식을 흉내"""
    N = X.shape[0]
    out = np.zeros((N, W.shape[1]))
    for i in range(N):
        out[i] = np.dot(X[i], W)
    return out


def vectorized_matmul(X, W):
    """배치 전체를 한 번에(벡터화) 처리 - GPU의 '대량 병렬 연산' 방식을 흉내"""
    return np.dot(X, W)


def compare_parallel_speedup(batch_size=2000, dim_in=784, dim_out=100, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(batch_size, dim_in)
    W = rng.randn(dim_in, dim_out)

    start = time.time()
    out_seq = sequential_matmul(X, W)
    seq_time = time.time() - start

    start = time.time()
    out_vec = vectorized_matmul(X, W)
    vec_time = time.time() - start

    assert np.allclose(out_seq, out_vec, atol=1e-8), '두 방식의 결과가 일치해야 함'

    print('=== 순차(for문) 연산 vs 벡터화(배치) 연산 : 처리 시간 비교 ===')
    print(f'배치 크기 {batch_size}, 입력 차원 {dim_in}, 출력 차원 {dim_out}')
    print(f'순차 처리 시간   : {seq_time:.4f}초')
    print(f'벡터화 처리 시간 : {vec_time:.4f}초')
    print(f'속도 향상 배율   : {seq_time / vec_time:.1f}배')
    print('-> 결과값은 완전히 동일하지만, 대량의 단순 곱셈-누적을 한꺼번에(병렬로) 처리하는 쪽이 훨씬 빠름')
    print('-> GPU가 CPU보다 딥러닝에 유리한 이유도 이와 같은 원리(대량 병렬 연산에 특화)')
    return seq_time, vec_time


# ---------------------------------------------------------------------------
# 8.3.3 분산 학습 : 미니배치를 여러 "워커"로 나눠 각자 기울기를 계산한 뒤 평균 낸 것이,
#   전체 배치를 한 번에 계산한 기울기와 (수치오차 범위 내에서) 같음을 확인 -> 데이터 병렬 분산 학습의 원리 검증
# ---------------------------------------------------------------------------
def compute_gradient(X, y, W):
    """단순 선형 회귀(Affine, bias 없음) : loss = mean(0.5 * (Xw - y)^2), dL/dW = (1/N) X^T (Xw - y)"""
    N = X.shape[0]
    y_pred = np.dot(X, W)
    error = y_pred - y
    grad = np.dot(X.T, error) / N
    return grad


def simulate_distributed_training(num_workers=4, samples_per_worker=50, dim=20, seed=0):
    rng = np.random.RandomState(seed)
    N = num_workers * samples_per_worker
    X = rng.randn(N, dim)
    W = rng.randn(dim, 1) * 0.1
    y = rng.randn(N, 1)

    # 방법 1 : 전체 배치를 한 번에(단일 장치라고 가정) 계산한 기울기
    full_grad = compute_gradient(X, y, W)

    # 방법 2 : 배치를 num_workers개로 나눠(여러 GPU/컴퓨터에 분산했다고 가정) 각자 기울기 계산 후 평균
    X_shards = np.split(X, num_workers)
    y_shards = np.split(y, num_workers)
    worker_grads = [compute_gradient(Xk, yk, W) for Xk, yk in zip(X_shards, y_shards)]
    averaged_grad = np.mean(worker_grads, axis=0)

    diff = np.linalg.norm(full_grad - averaged_grad)

    print()
    print('=== 분산 학습(데이터 병렬) 시뮬레이션 : 전체 배치 기울기 vs 워커별 기울기 평균 ===')
    print(f'전체 샘플 수 {N}개를 워커 {num_workers}개에 {samples_per_worker}개씩 분산')
    print(f'전체 배치로 한 번에 계산한 기울기 노름     : {np.linalg.norm(full_grad):.6f}')
    print(f'워커별로 나눠 계산 후 평균 낸 기울기 노름   : {np.linalg.norm(averaged_grad):.6f}')
    print(f'두 기울기의 차이(노름)                      : {diff:.2e}')
    assert diff < 1e-8, '워커 수로 균등 분할했을 때 두 기울기는 수치오차 범위 내에서 일치해야 함'
    print('-> 배치를 워커(GPU/컴퓨터) 수만큼 나눠 각자 기울기를 구한 뒤 평균 내면,')
    print('   전체 배치를 한 번에 계산한 것과 (수치적으로) 동일한 기울기를 얻음')
    print('-> 이것이 데이터 병렬(data parallel) 분산 학습이 성립하는 원리 : 통신으로 기울기만 모아 평균 내면 됨')
    return full_grad, averaged_grad


# ---------------------------------------------------------------------------
# 8.3.4 연산 정밀도와 비트 줄이기 : float16(반정밀도)/float32(단정밀도)/float64(배정밀도)의 메모리·오차 비교
# ---------------------------------------------------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def forward_with_dtype(X, W1, b1, W2, b2, dtype):
    X = X.astype(dtype)
    W1, b1 = W1.astype(dtype), b1.astype(dtype)
    W2, b2 = W2.astype(dtype), b2.astype(dtype)

    h = sigmoid(np.dot(X, W1) + b1)
    out = np.dot(h, W2) + b2
    return out


def compare_precision(batch_size=100, dim_in=784, dim_hidden=50, dim_out=10, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(batch_size, dim_in)
    W1 = rng.randn(dim_in, dim_hidden) * 0.01
    b1 = np.zeros(dim_hidden)
    W2 = rng.randn(dim_hidden, dim_out) * 0.01
    b2 = np.zeros(dim_out)

    dtypes = [np.float64, np.float32, np.float16]
    outputs = {}
    mem_bytes = {}
    for dtype in dtypes:
        outputs[dtype] = forward_with_dtype(X, W1, b1, W2, b2, dtype)
        # 전체 가중치(W1, W2)가 각 정밀도로 저장될 때 차지하는 메모리(바이트)
        mem_bytes[dtype] = W1.astype(dtype).nbytes + W2.astype(dtype).nbytes

    baseline = outputs[np.float64]  # 배정밀도(float64) 결과를 기준값으로 사용

    print()
    print('=== 연산 정밀도(float64/float32/float16)별 메모리 사용량과 순전파 결과 오차 비교 ===')
    print(f'{"dtype":>10} | {"가중치 메모리(bytes)":>18} | {"float64 대비 상대오차":>20}')
    for dtype in dtypes:
        rel_error = np.linalg.norm(outputs[dtype] - baseline) / np.linalg.norm(baseline)
        name = np.dtype(dtype).name
        print(f'{name:>10} | {mem_bytes[dtype]:>18} | {rel_error:>20.2e}')

    print('-> float32, float16으로 비트 수를 줄여도 float64 대비 순전파 결과의 상대오차가 매우 작음')
    print('-> 반면 가중치가 차지하는 메모리는 float64 대비 float32는 절반, float16은 1/4 수준으로 줄어듦')
    print('-> 신경망은 입력의 작은 노이즈에도 결과가 크게 달라지지 않는 강건성이 있어, 낮은 정밀도로도 충분히 학습·추론 가능')
    return mem_bytes


def plot_memory_comparison(mem_bytes, save_path):
    names = [np.dtype(dtype).name for dtype in mem_bytes]
    values = list(mem_bytes.values())

    plt.figure(figsize=(6, 5))
    plt.bar(names, values, color=['#4C72B0', '#55A868', '#C44E52'])
    plt.ylabel('가중치(W1+W2) 메모리 사용량 (bytes)')
    plt.title('연산 정밀도(dtype)에 따른 가중치 메모리 사용량 비교')
    for i, v in enumerate(values):
        plt.text(i, v, f'{v}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f'저장됨 : {save_path}')


if __name__ == '__main__':
    compare_parallel_speedup()
    simulate_distributed_training()
    mem_bytes = compare_precision()

    save_path = os.path.join(os.path.dirname(__file__), 'precision_memory.png')
    plot_memory_comparison(mem_bytes, save_path)
