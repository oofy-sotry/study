import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dataset.mnist import load_mnist


def relu(x):
    return np.maximum(0, x)

def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t, y = t.reshape(1, -1), y.reshape(1, -1)
    return -np.sum(t * np.log(y + 1e-7)) / y.shape[0]


# =============================================
# 6.5.1 검증 데이터 (Validation Data)
# 훈련 데이터의 일부를 분리해 하이퍼파라미터 조정에 사용
# 데이터 역할 구분
#   훈련 데이터 : 매개변수(가중치, 편향) 학습
#   검증 데이터 : 하이퍼파라미터 조정
#   시험 데이터 : 최종 성능 평가 전용 (학습·조정에 절대 사용 금지)
# =============================================
np.random.seed(0)
x_train_all, t_train_all, x_test, t_test = load_mnist(normalize=True, one_hot_label=True)

# 훈련 데이터를 섞은 뒤 앞 1000개를 검증용으로 분리
shuffle_idx = np.random.permutation(len(x_train_all))
x_train_all = x_train_all[shuffle_idx]
t_train_all = t_train_all[shuffle_idx]

val_size = 1000
x_val   = x_train_all[:val_size]
t_val   = t_train_all[:val_size]
x_train = x_train_all[val_size:val_size + 5000]   # 탐색 속도를 위해 5000개 사용
t_train = t_train_all[val_size:val_size + 5000]

print(f"훈련 데이터: {x_train.shape[0]}개  /  검증 데이터: {x_val.shape[0]}개  /  시험 데이터: {x_test.shape[0]}개")


# =============================================
# 6.5.2 하이퍼파라미터 최적화 — 랜덤 서치
# 탐색 대상:
#   lr (학습률)       : 로그 스케일 10^-3 ~ 10^0
#   λ  (가중치 감소)  : 로그 스케일 10^-8 ~ 10^-4
# 각 조합을 짧게 학습(10 에폭) 후 검증 정확도로 비교
# =============================================
def train_and_evaluate(x_tr, t_tr, x_val, t_val, lr, weight_decay, epochs=10):
    batch_size  = 100
    hidden_size = 100

    # He 초기화
    W1 = np.random.randn(x_tr.shape[1], hidden_size) * np.sqrt(2.0 / x_tr.shape[1])
    b1 = np.zeros(hidden_size)
    W2 = np.random.randn(hidden_size, t_tr.shape[1]) * np.sqrt(2.0 / hidden_size)
    b2 = np.zeros(t_tr.shape[1])

    for _ in range(epochs):
        idx = np.random.permutation(len(x_tr))
        for start in range(0, len(x_tr), batch_size):
            xb = x_tr[idx[start:start+batch_size]]
            tb = t_tr[idx[start:start+batch_size]]
            N  = xb.shape[0]

            # 순전파
            a1 = xb @ W1 + b1
            h1 = relu(a1)
            a2 = h1  @ W2 + b2
            y  = softmax(a2)

            # 역전파
            dy  = (y - tb) / N
            dW2 = h1.T @ dy
            db2 = dy.sum(axis=0)
            da1 = (dy @ W2.T) * (a1 > 0)
            dW1 = xb.T @ da1
            db1 = da1.sum(axis=0)

            # L2 가중치 감소 항 추가 : ∂L_new/∂W = ∂L/∂W + λW
            dW1 += weight_decay * W1
            dW2 += weight_decay * W2

            # SGD 갱신
            W1 -= lr * dW1;  b1 -= lr * db1
            W2 -= lr * dW2;  b2 -= lr * db2

    # 검증 정확도
    h1_v    = relu(x_val @ W1 + b1)
    pred    = np.argmax(softmax(h1_v @ W2 + b2), axis=1)
    val_acc = np.mean(pred == np.argmax(t_val, axis=1))
    return val_acc


search_count = 20
results      = []

print()
print("=" * 58)
print("하이퍼파라미터 랜덤 서치  (20회 시도, 각 10 에폭 학습)")
print("=" * 58)
print(f"{'시도':>4}  {'lr':>10}  {'λ':>12}  {'검증 정확도':>10}")
print("-" * 46)

for trial in range(search_count):
    lr           = 10 ** np.random.uniform(-3, 0)    # 10^-3 ~ 10^0
    weight_decay = 10 ** np.random.uniform(-8, -4)   # 10^-8 ~ 10^-4

    val_acc = train_and_evaluate(x_train, t_train, x_val, t_val, lr, weight_decay)
    results.append((val_acc, lr, weight_decay))
    print(f"  {trial+1:>2}  {lr:>10.6f}  {weight_decay:>12.2e}  {val_acc:>10.4f}")

# 검증 정확도 기준 내림차순 정렬
results.sort(reverse=True)

print()
print("=" * 58)
print("검증 정확도 상위 5개")
print("=" * 58)
print(f"{'순위':>4}  {'lr':>10}  {'λ':>12}  {'검증 정확도':>10}")
print("-" * 46)
for rank, (val_acc, lr, wd) in enumerate(results[:5], 1):
    print(f"  {rank:>2}  {lr:>10.6f}  {wd:>12.2e}  {val_acc:>10.4f}")

best_lrs = [r[1] for r in results[:5]]
best_wds = [r[2] for r in results[:5]]
print()
print(f"좋은 결과를 낸 lr 범위 : {min(best_lrs):.2e} ~ {max(best_lrs):.2e}")
print(f"좋은 결과를 낸 λ 범위  : {min(best_wds):.2e} ~ {max(best_wds):.2e}")
print("=> 이 범위를 좁혀 다음 랜덤 서치를 반복하면 최적 하이퍼파라미터로 수렴")
