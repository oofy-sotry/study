# coding: utf-8
"""NSMC 감성분류기 학습 스크립트 : 데이터 로드 -> vocab 구축 -> 학습/검증 분리 -> 학습 루프 -> 체크포인트 저장"""
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_utils import load_nsmc, build_vocab, save_vocab, NSMCDataset
from model import LSTMSentimentClassifier

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')
CKPT_DIR = os.path.join(BASE_DIR, 'checkpoints')

MAX_LEN = 60
BATCH_SIZE = 128
EPOCHS = 5
LR = 1e-3
VAL_RATIO = 0.1


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')  # Apple Silicon GPU
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def split_train_val(texts, labels, val_ratio, seed=0):
    """학습 데이터 중 일부를 검증용으로 떼어놓음 - 학습 도중 '한 번도 안 본 데이터'에서의 성능을 매 epoch마다 확인하기 위함
    (4장에서 배운 '훈련 데이터로 정확도를 재면 오버피팅을 못 알아챈다'는 것과 같은 이유)"""
    g = torch.Generator().manual_seed(seed)
    n = len(texts)
    perm = torch.randperm(n, generator=g).tolist()
    n_val = int(n * val_ratio)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_texts = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    return train_texts, train_labels, val_texts, val_labels


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for input_ids, lengths, labels in loader:
            input_ids, lengths, labels = input_ids.to(device), lengths.to(device), labels.to(device)

            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            preds = (torch.sigmoid(logits) >= 0.5).float()
            total_correct += (preds == labels).sum().item()
            total_count += labels.size(0)
            total_loss += loss.item() * labels.size(0)

    return total_loss / total_count, total_correct / total_count


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    device = get_device()
    print(f'사용 장치 : {device}')

    print('데이터 로드 중...')
    train_texts, train_labels = load_nsmc(os.path.join(DATA_DIR, 'ratings_train.txt'))
    print(f'전체 학습 리뷰 {len(train_texts)}개')

    print('어휘사전(vocab) 구축 중...')
    vocab = build_vocab(train_texts, max_vocab_size=20000, min_freq=2)
    save_vocab(vocab, os.path.join(CKPT_DIR, 'vocab.json'))
    print(f'vocab 크기 {len(vocab)} -> {os.path.join(CKPT_DIR, "vocab.json")} 에 저장')

    train_texts, train_labels, val_texts, val_labels = split_train_val(train_texts, train_labels, VAL_RATIO)
    print(f'학습 {len(train_texts)}개 / 검증 {len(val_texts)}개로 분리 (validation ratio={VAL_RATIO})')

    train_ds = NSMCDataset(train_texts, train_labels, vocab, max_len=MAX_LEN)
    val_ds = NSMCDataset(val_texts, val_labels, vocab, max_len=MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMSentimentClassifier(vocab_size=len(vocab)).to(device)
    criterion = nn.BCEWithLogitsLoss()  # sigmoid + 이진 교차 엔트로피를 한 번에 안정적으로 계산
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # 6장에서 다룬 Adam - 실전에서도 기본 선택지

    best_val_acc = 0.0
    print()
    print(f'=== 학습 시작 (epochs={EPOCHS}, batch_size={BATCH_SIZE}, lr={LR}) ===')
    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        elapsed = time.time() - start

        print(f'[epoch {epoch}/{EPOCHS}] '
              f'train_loss={train_loss:.4f} train_acc={train_acc:.4f} | '
              f'val_loss={val_loss:.4f} val_acc={val_acc:.4f} | {elapsed:.1f}초')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, 'best_model.pt'))
            print(f'  -> 검증 정확도 갱신, 체크포인트 저장 (val_acc={val_acc:.4f})')

    print()
    print(f'학습 종료. 최고 검증 정확도 : {best_val_acc:.4f}')


if __name__ == '__main__':
    main()
