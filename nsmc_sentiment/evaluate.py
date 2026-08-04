# coding: utf-8
"""학습된 best_model.pt를 테스트셋(ratings_test.txt, 5만 개)에 적용해 최종 정확도 평가"""
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_utils import load_nsmc, load_vocab, NSMCDataset
from model import LSTMSentimentClassifier
from train import get_device, run_epoch, MAX_LEN, BATCH_SIZE

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')
CKPT_DIR = os.path.join(BASE_DIR, 'checkpoints')


def main():
    device = get_device()
    print(f'사용 장치 : {device}')

    vocab = load_vocab(os.path.join(CKPT_DIR, 'vocab.json'))
    print(f'vocab 로드 완료 (크기 {len(vocab)})')

    test_texts, test_labels = load_nsmc(os.path.join(DATA_DIR, 'ratings_test.txt'))
    print(f'테스트 리뷰 {len(test_texts)}개 로드')

    test_ds = NSMCDataset(test_texts, test_labels, vocab, max_len=MAX_LEN)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMSentimentClassifier(vocab_size=len(vocab)).to(device)
    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, 'best_model.pt'), map_location=device))
    print('학습된 가중치(best_model.pt) 로드 완료')

    criterion = nn.BCEWithLogitsLoss()
    # optimizer는 eval 모드(train=False)에서는 쓰이지 않지만 run_epoch 시그니처를 맞추기 위해 형식상 전달
    dummy_optimizer = torch.optim.Adam(model.parameters())

    test_loss, test_acc = run_epoch(model, test_loader, criterion, dummy_optimizer, device, train=False)

    print()
    print('=== 테스트셋(한 번도 학습에 쓰이지 않은 5만 개) 최종 평가 ===')
    print(f'test_loss = {test_loss:.4f}')
    print(f'test_acc  = {test_acc:.4f}')


if __name__ == '__main__':
    main()
