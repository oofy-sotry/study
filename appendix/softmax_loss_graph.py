# coding: utf-8
"""부록 Softmax-with-Loss 계층의 계산 그래프 - exp/sum/(1/x)/mul/log/x(-1) 노드를 하나씩 구현해
역전파가 정확히 (y-t) 형태로 정리됨을 직접 확인 (5장에서 '결과만' 가져다 썼던 공식의 유도 검증)"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np


# ---------------------------------------------------------------------------
# 계산 그래프를 구성하는 기본 노드들 (각 노드는 순전파에서 쓴 값을 캐시해뒀다가 역전파에 사용)
# ---------------------------------------------------------------------------
class ExpNode:
    def forward(self, x):
        self.y = np.exp(x)
        return self.y

    def backward(self, dout):
        return dout * self.y  # d(exp(x))/dx = exp(x)


class SumNode:
    """여러 입력을 더하는 노드 - 역전파 시 상류 기울기를 모든 입력 갈래에 그대로 분배(복제)"""
    def forward(self, x):
        self.x = x
        return np.sum(x)

    def backward(self, dout):
        return dout * np.ones_like(self.x)


class ReciprocalNode:
    """"1/x" 노드"""
    def forward(self, x):
        self.x = x
        return 1.0 / x

    def backward(self, dout):
        return dout * (-1.0 / (self.x ** 2))  # d(1/x)/dx = -1/x^2


class MulNode:
    def forward(self, x, y):
        self.x, self.y = x, y
        return x * y

    def backward(self, dout):
        return dout * self.y, dout * self.x  # 순전파 때의 '다른 쪽 입력'을 상류 기울기에 곱함


class LogNode:
    def forward(self, x):
        self.x = x
        return np.log(x)

    def backward(self, dout):
        return dout * (1.0 / self.x)  # d(log(x))/dx = 1/x


# ---------------------------------------------------------------------------
# 위 노드들을 이어붙여 Softmax-with-Loss 계층을 "계산 그래프 그대로" 재현 (샘플 1개, a: (C,), t: (C,) 원-핫)
# ---------------------------------------------------------------------------
def softmax_with_loss_graph(a, t):
    C = a.shape[0]

    # --- Softmax 계층 순전파 ---
    exp_nodes = [ExpNode() for _ in range(C)]
    e = np.array([exp_nodes[k].forward(a[k]) for k in range(C)])

    sum_node1 = SumNode()
    S = sum_node1.forward(e)

    recip_node = ReciprocalNode()
    inv_S = recip_node.forward(S)

    mul_nodes1 = [MulNode() for _ in range(C)]
    y = np.array([mul_nodes1[k].forward(e[k], inv_S) for k in range(C)])

    # --- Cross Entropy Error 계층 순전파 ---
    log_nodes = [LogNode() for _ in range(C)]
    logy = np.array([log_nodes[k].forward(y[k]) for k in range(C)])

    mul_nodes2 = [MulNode() for _ in range(C)]
    tlogy = np.array([mul_nodes2[k].forward(t[k], logy[k]) for k in range(C)])

    sum_node2 = SumNode()
    total = sum_node2.forward(tlogy)
    L = -1.0 * total

    # --- Cross Entropy Error 계층 역전파 : dL=1 에서 시작 ---
    dtotal = 1.0 * (-1.0)                       # "x(-1)" 노드
    dtlogy = sum_node2.backward(dtotal)         # sum 노드 : 모든 갈래로 그대로 분배

    dlogy = np.zeros(C)
    for k in range(C):
        _dt_k, dlogy[k] = mul_nodes2[k].backward(dtlogy[k])  # mul 노드 (tk 쪽 기울기는 사용하지 않음)

    dy = np.array([log_nodes[k].backward(dlogy[k]) for k in range(C)])  # log 노드

    # --- Softmax 계층 역전파 ---
    de_direct = np.zeros(C)
    d_invS = 0.0
    for k in range(C):
        de_direct[k], dinvS_k = mul_nodes1[k].backward(dy[k])  # mul 노드 : exp(ak) 경로와 1/S 경로로 분기
        d_invS += dinvS_k                                       # 1/S는 C개 갈래가 공유하는 노드이므로 기울기를 모두 더함

    dS = recip_node.backward(d_invS)            # "1/x" 노드
    de_viaS = sum_node1.backward(dS)            # sum 노드 : 다시 모든 exp(ak) 갈래로 그대로 분배

    de = de_direct + de_viaS                    # exp(ak)로 흘러드는 두 경로(직접/합)의 기울기를 합침
    da = np.array([exp_nodes[k].backward(de[k]) for k in range(C)])  # exp 노드

    return y, L, da


# ---------------------------------------------------------------------------
# 검증 1 : 계산 그래프로 유도한 da 가 5장에서 쓴 단순화 공식 (y - t) 와 정확히 일치하는지 확인
# ---------------------------------------------------------------------------
def verify_against_simplified_formula(seed=0):
    rng = np.random.RandomState(seed)
    C = 4

    print('=== 검증 1 : 계산 그래프 노드별 역전파(da) vs 단순화 공식(y-t) 비교 ===')
    for trial in range(3):
        a = rng.randn(C)
        t = np.zeros(C)
        t[rng.randint(C)] = 1.0  # 원-핫 정답 레이블

        y, L, da_graph = softmax_with_loss_graph(a, t)
        da_simplified = y - t

        diff = np.max(np.abs(da_graph - da_simplified))
        print(f'trial {trial} : a={np.round(a, 3)}, t={t.astype(int)}')
        print(f'  손실 L = {L:.6f}')
        print(f'  계산 그래프로 구한 da   = {np.round(da_graph, 6)}')
        print(f'  단순화 공식(y-t)        = {np.round(da_simplified, 6)}')
        print(f'  최대 차이               = {diff:.2e}')
        assert diff < 1e-10, '계산 그래프 결과와 단순화 공식(y-t)은 정확히 일치해야 함'
    print('-> 노드 단위로 유도한 역전파 결과가 5장에서 그대로 가져다 쓴 (y-t) 공식과 완전히 일치함을 확인')


# ---------------------------------------------------------------------------
# 검증 2 : 배치 단위로 확장 - 계산 그래프 결과의 배치 평균이 (Y-T)/N 및 수치 미분(중심차분)과도 일치하는지 확인
# ---------------------------------------------------------------------------
def batch_loss(A, T):
    """배치 전체의 평균 교차 엔트로피 손실 (수치 미분 기준값 계산용)"""
    N = A.shape[0]
    exp_A = np.exp(A - np.max(A, axis=1, keepdims=True))  # 오버플로 방지용 shift (softmax 계산 시 흔히 쓰는 안정화 기법)
    Y = exp_A / np.sum(exp_A, axis=1, keepdims=True)
    return -np.sum(T * np.log(Y + 1e-12)) / N


def numerical_gradient(A, T, h=1e-4):
    grad = np.zeros_like(A)
    it = np.nditer(A, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        orig = A[idx]

        A[idx] = orig + h
        loss_plus = batch_loss(A, T)
        A[idx] = orig - h
        loss_minus = batch_loss(A, T)
        A[idx] = orig

        grad[idx] = (loss_plus - loss_minus) / (2 * h)
        it.iternext()
    return grad


def verify_batch_against_numerical_gradient(seed=1):
    rng = np.random.RandomState(seed)
    N, C = 3, 4

    A = rng.randn(N, C)
    T = np.zeros((N, C))
    for i in range(N):
        T[i, rng.randint(C)] = 1.0

    # 계산 그래프 방식 : 샘플마다 자신의 손실(L_i)에 대한 노드별 역전파 da_i = y_i - t_i 를 구한 뒤
    # batch_loss = (1/N) * sum_i(L_i) 이므로, 배치 평균 손실 기준 기울기는 샘플별로 1/N씩 스케일링한 것과 같음
    # (주의 : 서로 다른 샘플 a_i는 독립된 변수이므로 샘플들끼리 평균을 낼 대상이 아니라, 행(row)별로 그대로 비교해야 함)
    da_graph_rows = np.array([softmax_with_loss_graph(A[i], T[i])[2] for i in range(N)])
    da_graph_full = da_graph_rows / N

    # 수치 미분(중심 차분)으로 구한 배치 평균 손실의 기울기 (A 전체에 대한 (N, C) 기울기 행렬)
    da_numerical = numerical_gradient(A.copy(), T)

    diff = np.max(np.abs(da_graph_full - da_numerical))

    print()
    print('=== 검증 2 : 배치 평균 손실 기준 - 계산 그래프(각 행을 1/N) vs 수치 미분(중심차분) 비교 ===')
    print(f'계산 그래프 기울기(행렬)\n{np.round(da_graph_full, 6)}')
    print(f'수치 미분 기울기(행렬)\n{np.round(da_numerical, 6)}')
    print(f'최대 차이               = {diff:.2e}')
    assert diff < 1e-4, '계산 그래프 결과와 수치 미분 결과는 1e-4 이내로 일치해야 함'
    print('-> 노드 단위 계산 그래프로 유도한 기울기(샘플별 (y-t)를 배치 크기 N으로 나눈 값)가,')
    print('   손실 함수를 직접 수치 미분한 결과와도 일치함 (5장 SoftmaxWithLoss의 (y-t)/N 구현과 동일한 결론)')


if __name__ == '__main__':
    verify_against_simplified_formula()
    verify_batch_against_numerical_gradient()
