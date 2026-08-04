# coding: utf-8
"""NSMC 데이터 로딩 + 토큰화 + 어휘사전(vocab) + PyTorch Dataset"""
import re
import json
from collections import Counter

import torch
from torch.utils.data import Dataset

PAD_TOKEN, UNK_TOKEN = '<pad>', '<unk>'
PAD_ID, UNK_ID = 0, 1

# 한글 음절, 영문/숫자, 나머지 기호를 각각 하나의 토큰 단위로 분리 (형태소 분석기 없이 쓰는 간단한 규칙 기반 토크나이저)
TOKEN_PATTERN = re.compile(r'[가-힣]+|[a-zA-Z]+|[0-9]+|[^\sㄱ-힣a-zA-Z0-9]')


def tokenize(text):
    return TOKEN_PATTERN.findall(text)


def load_nsmc(path):
    """탭으로 구분된 NSMC 파일(id, document, label)을 읽어 (텍스트, 라벨) 리스트로 반환. 빈 리뷰는 제외"""
    texts, labels = [], []
    with open(path, encoding='utf-8') as f:
        next(f)  # 헤더 스킵
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) != 3 or not parts[1].strip():
                continue
            _id, document, label = parts
            texts.append(document)
            labels.append(int(label))
    return texts, labels


def build_vocab(texts, max_vocab_size=20000, min_freq=2):
    """학습 데이터의 토큰 빈도를 세어 상위 max_vocab_size개로 어휘사전 구성 (<pad>=0, <unk>=1 고정)"""
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    vocab = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
    for token, freq in counter.most_common():
        if freq < min_freq:
            break
        if len(vocab) >= max_vocab_size:
            break
        vocab[token] = len(vocab)
    return vocab


def save_vocab(vocab, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False)


def load_vocab(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def encode(text, vocab, max_len):
    """텍스트 -> 토큰 id 시퀀스. max_len보다 길면 자르고, 짧으면 <pad>(0)로 채움"""
    ids = [vocab.get(tok, UNK_ID) for tok in tokenize(text)][:max_len]
    length = len(ids)
    ids = ids + [PAD_ID] * (max_len - length)
    return ids, max(length, 1)  # length=0(빈 시퀀스)이면 LSTM에 넣을 수 없으므로 최소 1로 보정


class NSMCDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=60):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids, length = encode(self.texts[idx], self.vocab, self.max_len)
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


if __name__ == '__main__':
    import os
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    texts, labels = load_nsmc(os.path.join(data_dir, 'ratings_train.txt'))
    print(f'학습 리뷰 {len(texts)}개 로드')
    print('토큰화 예시 :', tokenize(texts[0]), '-> label', labels[0])

    vocab = build_vocab(texts)
    print(f'어휘사전 크기 : {len(vocab)} (min_freq=2, max_vocab_size=20000)')
    ids, length = encode(texts[0], vocab, max_len=60)
    print('인코딩 예시 :', ids[:length], f'(length={length})')
