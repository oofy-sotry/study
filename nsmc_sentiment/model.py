# coding: utf-8
"""Embedding + LSTM 기반 감성분류기 (이진 분류 : 긍정/부정)"""
import torch
import torch.nn as nn


class LSTMSentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=1, dropout=0.3, pad_id=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)  # 출력 1개(로짓) -> sigmoid로 긍정 확률 계산

    def forward(self, input_ids, lengths):
        embedded = self.embedding(input_ids)  # (batch, seq_len, embed_dim)

        # 패딩 부분은 LSTM 연산에서 제외하도록 pack (7장에서 배운 '낭비 없이 처리하기'와 같은 맥락의 최적화)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False,
        )
        _, (h_n, _) = self.lstm(packed)  # h_n : (num_layers, batch, hidden_dim) - 마지막 시점의 은닉 상태
        last_hidden = h_n[-1]  # 마지막 층의 최종 은닉 상태 -> 문장 전체를 요약한 벡터로 사용

        out = self.dropout(last_hidden)
        logit = self.fc(out).squeeze(-1)  # (batch,)
        return logit


if __name__ == '__main__':
    model = LSTMSentimentClassifier(vocab_size=100, embed_dim=8, hidden_dim=16)
    x = torch.randint(0, 100, (4, 10))
    lengths = torch.tensor([10, 7, 3, 1])
    logit = model(x, lengths)
    print('출력 shape :', logit.shape)  # (4,) 이어야 함
    print('시그모이드 확률 :', torch.sigmoid(logit))
