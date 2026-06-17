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
# 6.4.3 드롭아웃 — Inverted Dropout 방식
# 학습 시: 마스크 후 1/(1-drop_ratio)배로 스케일 보정
#   -> 드롭된 뉴런의 손실분을 살아남은 뉴런이 보상
#   -> 활성화값의 기댓값을 드롭아웃 없을 때와 동일하게 유지
# 추론 시: 전체 뉴런을 그대로 사용 (스케일 조정 불필요)
# backward: 드롭된 뉴런(mask=0)에는 기울기 0, 활성 뉴런은 1/(1-p) 스케일
# =============================================
class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask / (1.0 - self.dropout_ratio)
        else:
            return x  # 추론 시 스케일 조정 불필요

    def backward(self, dout):
        return dout * self.mask / (1.0 - self.dropout_ratio)


np.random.seed(42)
x_train, t_train, x_test, t_test = load_mnist(normalize=True, one_hot_label=True)

# 훈련 데이터 300개로 제한 — 과소 데이터로 오버피팅 유발
x_train, t_train = x_train[:300], t_train[:300]

input_size    = 784
hidden_size   = 100
output_size   = 10
hidden_layers = 4
lr            = 0.01
batch_size    = 100
epochs        = 201
eval_interval = 20    # 20 에폭마다 정확도 측정


def run_experiment(use_dropout, drop_ratio=0.5):
    params = {}
    sizes  = [input_size] + [hidden_size] * hidden_layers + [output_size]
    for i in range(len(sizes) - 1):
        params[f'W{i}'] = np.random.randn(sizes[i], sizes[i+1]) * np.sqrt(2.0 / sizes[i])
        params[f'b{i}'] = np.zeros(sizes[i+1])

    dropout_layers = [Dropout(drop_ratio) for _ in range(hidden_layers)] if use_dropout else None
    acc_log = []  # (epoch, train_acc, test_acc)

    for epoch in range(epochs):
        idx = np.random.permutation(len(x_train))
        for start in range(0, len(x_train), batch_size):
            xb = x_train[idx[start:start+batch_size]]
            tb = t_train[idx[start:start+batch_size]]

            # 순전파
            a      = xb
            caches = []
            for i in range(hidden_layers):
                z      = a @ params[f'W{i}'] + params[f'b{i}']
                a_prev = a
                a      = relu(z)
                if use_dropout:
                    a = dropout_layers[i].forward(a, train_flg=True)
                caches.append((a_prev, z, a))

            out = a @ params[f'W{hidden_layers}'] + params[f'b{hidden_layers}']
            y   = softmax(out)

            # 역전파
            da = (y - tb) / batch_size
            params[f'W{hidden_layers}'] -= lr * (caches[-1][2].T @ da)
            params[f'b{hidden_layers}'] -= lr * da.sum(axis=0)

            for i in reversed(range(hidden_layers)):
                a_prev, z, _ = caches[i]
                dz = da @ params[f'W{i+1}'].T
                if use_dropout:
                    dz = dropout_layers[i].backward(dz)
                dz = dz * (z > 0)
                params[f'W{i}'] -= lr * (a_prev.T @ dz)
                params[f'b{i}'] -= lr * dz.sum(axis=0)
                da = dz

        # eval_interval 에폭마다 정확도 측정 (추론 모드)
        if epoch % eval_interval == 0:
            def infer(x):
                a = x
                for i in range(hidden_layers):
                    z = a @ params[f'W{i}'] + params[f'b{i}']
                    a = relu(z)
                    if use_dropout:
                        a = dropout_layers[i].forward(a, train_flg=False)
                out = a @ params[f'W{hidden_layers}'] + params[f'b{hidden_layers}']
                return np.argmax(softmax(out), axis=1)

            tr_acc = np.mean(infer(x_train) == np.argmax(t_train, axis=1))
            te_acc = np.mean(infer(x_test)  == np.argmax(t_test,  axis=1))
            acc_log.append((epoch, tr_acc, te_acc))

    return acc_log


print("=" * 64)
print(f"오버피팅 vs 드롭아웃 비교  (훈련 데이터 300개, {hidden_layers}층 신경망)")
print("=" * 64)

log_no = run_experiment(use_dropout=False)
log_do = run_experiment(use_dropout=True, drop_ratio=0.5)

print(f"\n{'epoch':>5}  {'드롭아웃 없음':^20}   {'드롭아웃(0.5)':^20}")
print(f"{'':>5}  {'훈련':>8}  {'시험':>8}   {'훈련':>8}  {'시험':>8}")
print("-" * 58)

for (ep, tr_no, te_no), (_, tr_do, te_do) in zip(log_no, log_do):
    marker = " ←" if (te_do - te_no) > 0.02 else ""
    print(f"  {ep:>3}  {tr_no:>8.4f}  {te_no:>8.4f}   {tr_do:>8.4f}  {te_do:>8.4f}{marker}")

_, tr_no_f, te_no_f = log_no[-1]
_, tr_do_f, te_do_f = log_do[-1]
gap_no = tr_no_f - te_no_f
gap_do = tr_do_f - te_do_f

print()
print(f"최종 훈련 정확도  드롭아웃 없음: {tr_no_f:.4f}  /  드롭아웃: {tr_do_f:.4f}")
print(f"최종 시험 정확도  드롭아웃 없음: {te_no_f:.4f}  /  드롭아웃: {te_do_f:.4f}")
print(f"훈련-시험 차이    드롭아웃 없음: {gap_no:.4f}  /  드롭아웃: {gap_do:.4f}")
print(f"=> 드롭아웃이 오버피팅을 {'억제함 (훈련-시험 차이 감소)' if gap_do < gap_no else '억제하지 못함'}")
